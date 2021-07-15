
#include "threepp/Canvas.hpp"

#include "threepp/math/Vector2.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include <optional>
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

    void addMouseListener(const std::shared_ptr<MouseListener> &listener) {
        if (!mouseListeners.count(listener->uuid)) {
            mouseListeners[listener->uuid] = listener;
        }
    }

    bool removeMouseListener(const std::string &listenerUuid) {
        if (mouseListeners.count(listenerUuid)) {
            mouseListeners.erase(listenerUuid);
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
    Vector2 lastMousePos{};
    std::vector<std::shared_ptr<KeyListener>> keyListeners;
    std::unordered_map<std::string, std::shared_ptr<MouseListener>> mouseListeners;
    std::optional<std::function<void(WindowSize)>> resizeListener;

    GLFWwindow *window;

    static void window_size_callback(GLFWwindow *w, int width, int height) {
        auto p = static_cast<Canvas::Impl *>(glfwGetWindowUserPointer(w));
        p->size_ = {width, height};
        if (p->resizeListener) p->resizeListener.value().operator()(p->size_);
    }

    static void error_callback(int error, const char *description) {
        fprintf(stderr, "Error: %s\n", description);
    }

    static void scroll_callback(GLFWwindow *w, double xoffset, double yoffset) {
        auto p = static_cast<Canvas::Impl *>(glfwGetWindowUserPointer(w));
        auto listeners = p->mouseListeners;
        if (listeners.empty()) return;
        Vector2 delta {(float) xoffset, (float) yoffset};
        for (auto &[_, l] :listeners) {
            l->onMouseWheel(delta);
        }
    }

    static void mouse_callback(GLFWwindow *w, int button, int action, int mods) {
        auto p = static_cast<Canvas::Impl *>(glfwGetWindowUserPointer(w));

        auto listeners = p->mouseListeners;
        for (auto &[_, l] : listeners) {

            switch (action) {
                case GLFW_PRESS:
                    l->onMouseDown(button, p->lastMousePos);
                    break;
                case GLFW_RELEASE:
                    l->onMouseUp(button, p->lastMousePos);
                    break;
                default:
                    break;
            }
        }
    }

    static void cursor_callback(GLFWwindow *w, double xpos, double ypos) {
        auto p = static_cast<Canvas::Impl *>(glfwGetWindowUserPointer(w));
        p->lastMousePos.set((float) xpos, (float) ypos);
        auto listeners = p->mouseListeners;
        for (auto &[_, l] : listeners) {
            l->onMouseMove(p->lastMousePos);
        }
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
                default:
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

void threepp::Canvas::addKeyAdapter(const KeyAdapter::Mode &mode, const std::function<void(KeyEvent)> &f) {
    addKeyListener(std::make_shared<KeyAdapter>(mode, f));
}

void threepp::Canvas::addMouseListener(const std::shared_ptr<MouseListener> &listener) {
    pimpl_->addMouseListener(listener);
}

bool threepp::Canvas::removeMouseListener(const std::string &listenerUuid) {
    return pimpl_->removeMouseListener(listenerUuid);
}

threepp::Canvas::~Canvas() = default;

Canvas::Parameters &threepp::Canvas::Parameters::title(std::string value) {

    this->title_ = std::move(value);

    return *this;
}

Canvas::Parameters &threepp::Canvas::Parameters::size(WindowSize size) {

    this->size_ = size;

    return *this;
}

Canvas::Parameters &threepp::Canvas::Parameters::size(int width, int height) {

    return this->size({width, height});
}

Canvas::Parameters &threepp::Canvas::Parameters::antialising(int antialiasing) {

    this->antialiasing_ = antialiasing;

    return *this;
}
