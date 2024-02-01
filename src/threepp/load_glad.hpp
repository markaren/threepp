
#ifndef THREEPP_LOAD_GLAD_HPP
#define THREEPP_LOAD_GLAD_HPP

#include "glad/glad.h"

#include <iostream>

static bool gladInitialized = false;

inline void initializeOpenGL() {

    if (!gladInitialized) {
        if (!gladLoadGL()) {
            std::cerr << "Failed to initialize GLAD" << std::endl;
            exit(EXIT_FAILURE);
        }
        gladInitialized = true;
    }
}

#endif//THREEPP_LOAD_GLAD_HPP
