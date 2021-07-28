
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include <iostream>

#include "threepp/renderers/gl/GLCapabilities.hpp"

int main() {

    glfwInit();
    auto window = glfwCreateWindow(640, 480, "Simple example", nullptr, nullptr);
    glfwMakeContextCurrent(window);

    gladLoadGL();

    std::cout << threepp::gl::GLCapabilities::instance() << std::endl;

    glfwDestroyWindow(window);

    glfwTerminate();

    return 0;
}
