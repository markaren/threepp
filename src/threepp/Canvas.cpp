
#include "threepp/Canvas.hpp"
#include "threepp/loaders/ImageLoader.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include <optional>
#include <unordered_map>
#include <queue>

using namespace threepp;

class Canvas::Impl {

public:
    GLFWwindow *window;

    WindowSize size_;
    Vector2 lastMousePos{};

    std::queue<std::function<void()>> tasks_;
    std::optional<std::function<void(WindowSize)>> resizeListener;
    std::unordered_map<std::string, std::shared_ptr<KeyListener>> keyListeners;
    std::unordered_map<std::string, std::shared_ptr<MouseListener>> mouseListeners;

    explicit Impl(const Canvas::Parameters &params) : size_(params.size_) {
        glfwSetErrorCallback(error_callback);

        if (!glfwInit()) {
            exit(EXIT_FAILURE);
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        if (params.antialiasing_ > 0) {
            glfwWindowHint(GLFW_SAMPLES, params.antialiasing_);
        }

        window = glfwCreateWindow(params.size_.width, params.size_.height, params.title_.c_str(), nullptr, nullptr);
        if (!window) {
            glfwTerminate();
            exit(EXIT_FAILURE);
        }

        {

            ImageLoader imageLoader;
            auto favicon = imageLoader.load("favicon.png", 4);
            if (favicon) {
                GLFWimage images[1];
                images[0] = {static_cast<int>(favicon->width),
                             static_cast<int>(favicon->height),
                             favicon->getData()};
                glfwSetWindowIcon(window, 1, images);
            }
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

    [[nodiscard]] const WindowSize& getSize() const {
        return size_;
    }

    void setSize(WindowSize size) const {
        glfwSetWindowSize(window, size.width, size.height);
    }

    void animate(const std::function<void()> &f) {
        while (!glfwWindowShouldClose(window)) {

            while (!tasks_.empty()) {
                auto task = tasks_.front();
                task();
                tasks_.pop();
            }

            f();

            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }

    void animate(const std::function<void(float)> &f) {
        Clock clock;
        while (!glfwWindowShouldClose(window)) {

            while (!tasks_.empty()) {
                auto task = tasks_.front();
                task();
                tasks_.pop();
            }

            f(clock.getDelta());

            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }

    void onWindowResize(std::function<void(WindowSize)> f) {
        this->resizeListener = std::move(f);
    }

    void addKeyListener(const std::shared_ptr<KeyListener> &listener) {
        if (!keyListeners.count(listener->uuid)) {
            keyListeners[listener->uuid] = listener;
        }
    }

    bool removeKeyListener(const std::string &listenerUuid) {
        if (keyListeners.count(listenerUuid)) {
            keyListeners.erase(listenerUuid);
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

    void invokeLater(const std::function<void()>& f) {
        tasks_.emplace(f);
    }

    ~Impl() {
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    static void window_size_callback(GLFWwindow *w, int width, int height) {
        auto p = static_cast<Canvas::Impl *>(glfwGetWindowUserPointer(w));
        p->size_ = {width, height};
        if (p->resizeListener) p->resizeListener.value().operator()(p->size_);
    }

    static void error_callback(int error, const char *description) {
        std::cerr << "Error: " << description << std::endl;
    }

    static void scroll_callback(GLFWwindow *w, double xoffset, double yoffset) {
        auto p = static_cast<Canvas::Impl *>(glfwGetWindowUserPointer(w));
        auto listeners = p->mouseListeners;
        if (listeners.empty()) return;
        Vector2 delta{(float) xoffset, (float) yoffset};
        for (auto &[_, l] : listeners) {
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
        p->lastMousePos.set(static_cast<float>(xpos), static_cast<float>(ypos));
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
        auto listeners = p->keyListeners;
        for (auto &[_, l] : listeners) {
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

void Canvas::animate(const std::function<void()> &f) {

    pimpl_->animate(f);
}

void Canvas::animate(const std::function<void(float)> &f) {

    pimpl_->animate(f);
}

const WindowSize& Canvas::getSize() const {

    return pimpl_->getSize();
}

float Canvas::getAspect() const {

    return getSize().getAspect();
}

void Canvas::setSize(WindowSize size) {

    pimpl_->setSize(size);
}
void Canvas::onWindowResize(std::function<void(WindowSize)> f) {

    pimpl_->onWindowResize(std::move(f));
}

void Canvas::addKeyListener(const std::shared_ptr<KeyListener> &listener) {

    pimpl_->addKeyListener(listener);
}

bool Canvas::removeKeyListener(const std::string &listenerUuid) {

    return pimpl_->removeKeyListener(listenerUuid);
}

std::string Canvas::addKeyAdapter(const KeyAdapter::Mode &mode, const std::function<void(KeyEvent)> &f) {

    auto listener = std::make_shared<KeyAdapter>(mode, f);
    addKeyListener(listener);
    return listener->uuid;
}

void Canvas::addMouseListener(const std::shared_ptr<MouseListener> &listener) {

    pimpl_->addMouseListener(listener);
}

bool Canvas::removeMouseListener(const std::string &listenerUuid) {

    return pimpl_->removeMouseListener(listenerUuid);
}

void Canvas::invokeLater(const std::function<void()>& f) {
    pimpl_->invokeLater(f);
}

Canvas::~Canvas() = default;

Canvas::Parameters &Canvas::Parameters::title(std::string value) {

    this->title_ = std::move(value);

    return *this;
}

Canvas::Parameters &Canvas::Parameters::size(WindowSize size) {

    this->size_ = size;

    return *this;
}

Canvas::Parameters &Canvas::Parameters::size(int width, int height) {

    return this->size({width, height});
}

Canvas::Parameters &Canvas::Parameters::antialiasing(int antialiasing) {

    this->antialiasing_ = antialiasing;

    return *this;
}

void *Canvas::window_ptr() const{

    return pimpl_->window;
}
