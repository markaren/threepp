

#include "threepp/utils/LoadGlad.hpp"

#include <iostream>


void threepp::loadGlad() {

    static bool gladInitialized = false;

    if (!gladInitialized) {
        if (!gladLoadGL()) {
            std::cerr << "Failed to initialize GLAD" << std::endl;
            exit(EXIT_FAILURE);
        }
        gladInitialized = true;
    }
}
