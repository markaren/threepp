
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

        if (params.antialiasing_ > 0) {
            glfwWindowHint(GLFW_SAMPLES, params.antialiasing_);
        }

        window = glfwCreateWindow(params.size_.width, params.size_.height, params.title_.c_str(), nullptr, nullptr);
        if (!window) {
            glfwTerminate();
            exit(EXIT_FAILURE);
        }

        glfwSetWindowUserPointer(window, this);

        glfwSetKeyCallback(window, key_callback);
        glfwSetMouseButtonCallback(window, mouse_callback);
        glfwSetCursorPosCallback(window, cursor_callback);
        glfwSetScrollCallback(window, scroll_callback);
        glfwSetWindowSizeCallback(window, window_size_callback);

        glfwMakeContextCurrent(window);
        gladLoadGL();
        glfwSwapInterval(1);

        if (params.antialiasing_ > 0) {
            glEnable(GL_MULTISAMPLE);
        }

        glEnable(GL_PROGRAM_POINT_SIZE);
        //        glEnable(GL_POINT_SPRITE);
        //        glEnable(GL_POINT_SMOOTH);
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

    void addKeyListener(const std::shared_ptr<KeyListener> &listener) {
        keyListeners.emplace_back(listener);
    }

    bool removeKeyListener(const std::shared_ptr<KeyListener> &listener) {
        auto find = std::find(keyListeners.begin(), keyListeners.end(), listener);
        if (find != keyListeners.end()) {
            keyListeners.erase(find);
            return true;
        }
        return false;
    }

    ~Impl() {
        glfwDestroyWindow(window);
        glfwTerminate();
    }

private:
    WindowSize size_;
    std::function<void(WindowSize)> resizeListener;
    std::vector<std::shared_ptr<KeyListener>> keyListeners;

    GLFWwindow *window;

    static void window_size_callback(GLFWwindow *w, int width, int height) {
        auto p = static_cast<Canvas::Impl *>(glfwGetWindowUserPointer(w));
        p->size_ = {width, height};
        p->resizeListener(p->size_);
    }

    static void error_callback(int error, const char *description) {
        fprintf(stderr, "Error: %s\n", description);
    }

    static void scroll_callback(GLFWwindow *w, double xoffset, double yoffset) {
        auto p = static_cast<Canvas::Impl *>(glfwGetWindowUserPointer(w));
    }

    static void mouse_callback(GLFWwindow *w, int button, int action, int mods) {
        auto p = static_cast<Canvas::Impl *>(glfwGetWindowUserPointer(w));

        switch (action) {
            case GLFW_PRESS:
                break;
            case GLFW_RELEASE:
                break;
        }
    }

    static void cursor_callback(GLFWwindow *w, double xpos, double ypos) {
        auto p = static_cast<Canvas::Impl *>(glfwGetWindowUserPointer(w));
    }

    static void key_callback(GLFWwindow *w, int key, int scancode, int action, int mods) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
            glfwSetWindowShouldClose(w, GLFW_TRUE);
            return;
        }

        auto p = static_cast<Canvas::Impl *>(glfwGetWindowUserPointer(w));
        if (p->keyListeners.empty()) return;

        KeyEvent evt{key, scancode, mods};
        for (auto &l : p->keyListeners) {
            switch (action) {
                case GLFW_PRESS:
                    l->onKeyPressed(evt);
                    break;
                case GLFW_RELEASE:
                    l->onKeyReleased(evt);
                    break;
                case GLFW_REPEAT:
                    l->onKeyRepeat(evt);
                    break;
            }
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

void threepp::Canvas::addKeyListener(const std::shared_ptr<KeyListener> &listener) {
    pimpl_->addKeyListener(listener);
}

bool threepp::Canvas::removeKeyListener(const std::shared_ptr<KeyListener> &listener) {
    return pimpl_->removeKeyListener(listener);
}

void threepp::Canvas::addKeyPressListener(const std::function<void(KeyEvent)> &f) {
    addKeyListener(std::make_shared<KeyAdapter>(KeyAdapter::KEY_PRESSED, f));
}

threepp::Canvas::~Canvas() = default;
