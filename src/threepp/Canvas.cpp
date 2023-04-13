
#include "threepp/Canvas.hpp"
#include "threepp/loaders/ImageLoader.hpp"

#include "threepp/core/Clock.hpp"
#include "threepp/utils/StringUtils.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include <iostream>
#include <optional>
#include <queue>

using namespace threepp;

namespace {

    typedef std::pair<std::function<void()>, float> task;

    struct CustomComparator {
        bool operator()(const task& l, const task& r) const { return l.second > r.second; }
    };

}// namespace

struct Canvas::Impl {

    GLFWwindow* window;

    WindowSize size_;
    Vector2 lastMousePos_;

    std::priority_queue<task, std::vector<task>, CustomComparator> tasks_;
    std::optional<std::function<void(WindowSize)>> resizeListener;
    std::vector<KeyListener*> keyListeners;
    std::vector<MouseListener*> mouseListeners;

    explicit Impl(const Canvas::Parameters& params): size_(params.size_) {
        glfwSetErrorCallback(error_callback);

        if (!glfwInit()) {
            exit(EXIT_FAILURE);
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
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
        glfwSwapInterval(params.vsync_ ? 1 : 0);

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

    inline void handleTasks() {
        while (!tasks_.empty()) {
            auto& task = tasks_.top();
            if (task.second < glfwGetTime()) {
                task.first();
                tasks_.pop();
            } else {
                break;
            }
        }
    }

    void animate(const std::function<void()>& f) {
        double lastTime = glfwGetTime();
        int nbFrames = 0;
        while (!glfwWindowShouldClose(window)) {

            handleTasks();

            f();

            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }

    void animate(const std::function<void(float)>& f) {

        double lastTime = glfwGetTime();
        int nbFrames = 0;
        Clock clock;
        while (!glfwWindowShouldClose(window)) {

            handleTasks();

            f(clock.getDelta());

            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }

    void animate(const std::function<void(float, float)>& f) {

        double lastTime = glfwGetTime();
        int nbFrames = 0;
        Clock clock;
        while (!glfwWindowShouldClose(window)) {

            handleTasks();

            f(static_cast<float>(glfwGetTime()), clock.getDelta());

            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }

    void onWindowResize(std::function<void(WindowSize)> f) {
        this->resizeListener = std::move(f);
    }

    void addKeyListener(KeyListener* listener) {
        auto find = std::find(keyListeners.begin(), keyListeners.end(), listener);
        if (find == keyListeners.end()) {
            keyListeners.emplace_back(listener);
        }
    }

    bool removeKeyListener(const KeyListener* listener) {
        auto find = std::find(keyListeners.begin(), keyListeners.end(), listener);
        if (find != keyListeners.end()) {
            keyListeners.erase(find);
            return true;
        }
        return false;
    }

    void addMouseListener(MouseListener* listener) {
        auto find = std::find(mouseListeners.begin(), mouseListeners.end(), listener);
        if (find == mouseListeners.end()) {
            mouseListeners.emplace_back(listener);
        }
    }

    bool removeMouseListener(const MouseListener* listener) {
        auto find = std::find(mouseListeners.begin(), mouseListeners.end(), listener);
        if (find != mouseListeners.end()) {
            mouseListeners.erase(find);
            return true;
        }
        return false;
    }

    void invokeLater(const std::function<void()>& f, float t) {
        tasks_.emplace(f, static_cast<float>(glfwGetTime()) + t);
    }

    ~Impl() {
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    static void window_size_callback(GLFWwindow* w, int width, int height) {
        auto p = static_cast<Canvas::Impl*>(glfwGetWindowUserPointer(w));
        p->size_ = {width, height};
        if (p->resizeListener) p->resizeListener.value().operator()(p->size_);
    }

    static void error_callback(int error, const char* description) {
        std::cerr << "Error: " << description << std::endl;
    }

    static void scroll_callback(GLFWwindow* w, double xoffset, double yoffset) {
        auto p = static_cast<Canvas::Impl*>(glfwGetWindowUserPointer(w));
        auto listeners = p->mouseListeners;
        if (listeners.empty()) return;
        Vector2 delta{(float) xoffset, (float) yoffset};
        for (auto l : listeners) {
            l->onMouseWheel(delta);
        }
    }

    static void mouse_callback(GLFWwindow* w, int button, int action, int mods) {
        auto p = static_cast<Canvas::Impl*>(glfwGetWindowUserPointer(w));

        auto listeners = p->mouseListeners;
        for (auto l : listeners) {

            switch (action) {
                case GLFW_PRESS:
                    l->onMouseDown(button, p->lastMousePos_);
                    break;
                case GLFW_RELEASE:
                    l->onMouseUp(button, p->lastMousePos_);
                    break;
                default:
                    break;
            }
        }
    }

    static void cursor_callback(GLFWwindow* w, double xpos, double ypos) {
        auto p = static_cast<Canvas::Impl*>(glfwGetWindowUserPointer(w));
        p->lastMousePos_.set(static_cast<float>(xpos), static_cast<float>(ypos));
        auto listeners = p->mouseListeners;
        for (auto l : listeners) {
            l->onMouseMove(p->lastMousePos_);
        }
    }

    static void key_callback(GLFWwindow* w, int key, int scancode, int action, int mods) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
            glfwSetWindowShouldClose(w, GLFW_TRUE);
            return;
        }

        auto p = static_cast<Canvas::Impl*>(glfwGetWindowUserPointer(w));
        if (p->keyListeners.empty()) return;

        KeyEvent evt{key, scancode, mods};
        auto listeners = p->keyListeners;
        for (auto l : listeners) {
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

Canvas::Canvas(const Canvas::Parameters& params)
    : pimpl_(new Impl(params)) {}

Canvas::Canvas(const std::string& name)
    : Canvas(Canvas::Parameters().title(name)) {}

Canvas::Canvas(const std::string& name, const std::unordered_map<std::string, ParameterValue>& values)
    : Canvas(Canvas::Parameters(values).title(name)) {}


void Canvas::animate(const std::function<void()>& f) {

    pimpl_->animate(f);
}

void Canvas::animate(const std::function<void(float)>& f) {

    pimpl_->animate(f);
}

void Canvas::animate(const std::function<void(float, float)>& f) {

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

void Canvas::addKeyListener(KeyListener* listener) {

    pimpl_->addKeyListener(listener);
}

bool Canvas::removeKeyListener(const KeyListener* listener) {

    return pimpl_->removeKeyListener(listener);
}

void Canvas::addMouseListener(MouseListener* listener) {

    pimpl_->addMouseListener(listener);
}

bool Canvas::removeMouseListener(const MouseListener* listener) {

    return pimpl_->removeMouseListener(listener);
}

void Canvas::invokeLater(const std::function<void()>& f, float t) {

    pimpl_->invokeLater(f, t);
}

void* Canvas::window_ptr() const {

    return pimpl_->window;
}

Canvas::~Canvas() = default;


Canvas::Parameters::Parameters() = default;

Canvas::Parameters& Canvas::Parameters::title(std::string value) {

    this->title_ = std::move(value);

    return *this;
}

Canvas::Parameters& Canvas::Parameters::size(WindowSize size) {

    this->size_ = size;

    return *this;
}

Canvas::Parameters& Canvas::Parameters::size(int width, int height) {

    return this->size({width, height});
}

Canvas::Parameters& Canvas::Parameters::antialiasing(int antialiasing) {

    this->antialiasing_ = antialiasing;

    return *this;
}

Canvas::Parameters& Canvas::Parameters::vsync(bool flag) {

    this->vsync_ = flag;

    return *this;
}

Canvas::Parameters::Parameters(const std::unordered_map<std::string, ParameterValue>& values) {

    std::vector<std::string> unused;
    for (const auto& [key, value] : values) {

        bool used = false;

        if (key == "antialiasing") {

            antialiasing(std::get<int>(value));
            used = true;

        } else if (key == "vsync") {

            vsync(std::get<bool>(value));
            used = true;

        } else if (key == "size") {

            auto _size = std::get<WindowSize>(value);
            size(_size);
            used = true;
        }

        if (!used) {
            unused.emplace_back(key);
        }
    }

    if (!unused.empty()) {

        std::cerr << "Unused Canvas parameters: [" << utils::join(unused, ',') << "]" << std::endl;

    }

}
