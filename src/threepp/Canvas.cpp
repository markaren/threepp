
#include "threepp/Canvas.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/glad.h>

using namespace threepp;

struct Canvas::Impl {

    GLFWwindow *window;

    explicit Impl(const Canvas::Parameters &params) {
        glfwSetErrorCallback(error_callback);

        if (!glfwInit()) {
            exit(EXIT_FAILURE);
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

        window = glfwCreateWindow(params.width_, params.height_, params.title_.c_str(), nullptr, nullptr);
        if (!window) {
            glfwTerminate();
            exit(EXIT_FAILURE);
        }

        glfwSetKeyCallback(window, key_callback);

        glfwMakeContextCurrent(window);
        gladLoadGL();
        glfwSwapInterval(1);
    }

    void animate(const std::function<void(float)> &f) const {
        Clock clock;
        while (!glfwWindowShouldClose(window)) {

            f(clock.getDelta());

            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }

    ~Impl() {
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    static void error_callback(int error, const char *description) {
        fprintf(stderr, "Error: %s\n", description);
    }

    static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
};

Canvas::Canvas(const Canvas::Parameters &params) : pimpl_(new Impl(params)) {}

void Canvas::animate(const std::function<void(float)> &f) const {

    pimpl_->animate(f);
}

Canvas::~Canvas() = default;
