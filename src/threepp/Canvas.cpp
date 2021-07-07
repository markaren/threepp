
#include "threepp/Canvas.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include <utility>

using namespace threepp;

class Canvas::Impl {

public:
    explicit Impl(const Canvas::Parameters &params) : size_(params.size_) {
        glfwSetErrorCallback(error_callback);

        if (!glfwInit()) {
            exit(EXIT_FAILURE);
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

        if (params.antialiasing > 0) {
            glfwWindowHint(GLFW_SAMPLES, params.antialiasing);
        }

        window = glfwCreateWindow(params.size_.width, params.size_.height, params.title_.c_str(), nullptr, nullptr);
        if (!window) {
            glfwTerminate();
            exit(EXIT_FAILURE);
        }

        glfwSetWindowUserPointer(window, this);

        glfwSetKeyCallback(window, key_callback);
        glfwSetWindowSizeCallback(window, window_size_callback);

        glfwMakeContextCurrent(window);
        gladLoadGL();
        glfwSwapInterval(1);

        if (params.antialiasing > 0) {
            glEnable(GL_MULTISAMPLE);
        }
    }

    [[nodiscard]] WindowSize getSize() const {
        return size_;
    }

    void setSize(WindowSize size) const {
        glfwSetWindowSize(window, size.width, size.height);
    }

    void animate(const std::function<void(float)> &f) const {
        Clock clock;
        while (!glfwWindowShouldClose(window)) {

            f(clock.getDelta());

            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }

    void onWindowResize(std::function<void(WindowSize)> f) {
        this->resizeListener = std::move(f);
    }

    ~Impl() {
        glfwDestroyWindow(window);
        glfwTerminate();
    }

private:
    WindowSize size_;
    std::function<void(WindowSize)> resizeListener;

    GLFWwindow *window;

    static void window_size_callback(GLFWwindow *w, int width, int height) {
        auto p = static_cast<Canvas::Impl *>(glfwGetWindowUserPointer(w));
        p->size_.width = width;
        p->size_.height = height;
        p->resizeListener(p->size_);
    }

    static void error_callback(int error, const char *description) {
        fprintf(stderr, "Error: %s\n", description);
    }

    static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
    }
};

Canvas::Canvas(const Canvas::Parameters &params) : pimpl_(new Impl(params)) {}

void Canvas::animate(const std::function<void(float)> &f) const {

    pimpl_->animate(f);
}

WindowSize threepp::Canvas::getSize() const {

    return pimpl_->getSize();
}

float threepp::Canvas::getAspect() const {
    return getSize().getAspect();
}

void threepp::Canvas::setSize(WindowSize size) {

    pimpl_->setSize(size);
}
void threepp::Canvas::onWindowResize(std::function<void(WindowSize)> f) {
    pimpl_->onWindowResize(std::move(f));
}

threepp::Canvas::~Canvas() = default;
