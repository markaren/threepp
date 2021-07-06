
#include "threepp/Canvas.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/glad.h>

using namespace threepp;

class Canvas::Impl {

public:
    explicit Impl(const Canvas::Parameters &params) : width_(params.width_), height_(params.height_) {
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
        glfwSetWindowSizeCallback(window, window_size_callback);

        glfwMakeContextCurrent(window);
        gladLoadGL();
        glfwSwapInterval(1);
    }

    [[nodiscard]] int getWidth() const {
        return width_;
    }

    void setSize(int width, int height) const {
        glfwSetWindowSize(window, width, height);
    }

    [[nodiscard]] int getHeight() const {
        return height_;
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

private:
    int width_;
    int height_;

    GLFWwindow *window;

    static void window_size_callback(GLFWwindow *, int width, int height) {
        //        this->width = width;
        //        this->height = height;
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

int threepp::Canvas::getWidth() const {

    return pimpl_->getWidth();
}

int threepp::Canvas::getHeight() const {

    return pimpl_->getHeight();
}

float threepp::Canvas::getAspect() const {

    return (float) getWidth() / (float) getWidth();
}

void threepp::Canvas::setSize(int width, int height) {

    pimpl_->setSize(width, height);
}

threepp::Canvas::~Canvas() = default;
