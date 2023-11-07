
#include "threepp/canvas/Canvas.hpp"

#include "threepp/favicon.hpp"
#include "threepp/loaders/ImageLoader.hpp"
#include "threepp/utils/StringUtils.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include <iostream>
#include <optional>
#include <queue>

using namespace threepp;

namespace {

    typedef std::pair<std::function<void()>, float> Task;

    struct CustomComparator {
        bool operator()(const Task& l, const Task& r) const { return l.second > r.second; }
    };

    void setWindowIcon(GLFWwindow* window, std::optional<std::filesystem::path> customIcon) {

        ImageLoader imageLoader;
        std::optional<Image> favicon;
        if (customIcon) {
            favicon = imageLoader.load(*customIcon, 4, false);
        } else {
            favicon = imageLoader.load(faviconSource(), 4, false);
        }
        if (favicon) {
            GLFWimage images[1];
            images[0] = {static_cast<int>(favicon->width),
                         static_cast<int>(favicon->height),
                         favicon->data().data()};
            glfwSetWindowIcon(window, 1, images);
        }
    }

    Key glfwKeyCodeToKey(int keyCode) {

        // clang-format off
        switch (keyCode) {
            case GLFW_KEY_0: return Key::NUM_0;
            case GLFW_KEY_1: return Key::NUM_1;
            case GLFW_KEY_2: return Key::NUM_2;
            case GLFW_KEY_3: return Key::NUM_3;
            case GLFW_KEY_4: return Key::NUM_4;
            case GLFW_KEY_5: return Key::NUM_5;
            case GLFW_KEY_6: return Key::NUM_6;
            case GLFW_KEY_7: return Key::NUM_7;
            case GLFW_KEY_8: return Key::NUM_8;
            case GLFW_KEY_9: return Key::NUM_9;

            case GLFW_KEY_F1: return Key::F1;
            case GLFW_KEY_F2: return Key::F2;
            case GLFW_KEY_F3: return Key::F3;
            case GLFW_KEY_F4: return Key::F4;
            case GLFW_KEY_F5: return Key::F5;
            case GLFW_KEY_F6: return Key::F6;
            case GLFW_KEY_F7: return Key::F7;
            case GLFW_KEY_F8: return Key::F8;
            case GLFW_KEY_F9: return Key::F9;
            case GLFW_KEY_F10: return Key::F10;
            case GLFW_KEY_F11: return Key::F11;
            case GLFW_KEY_F12: return Key::F12;

            case GLFW_KEY_A: return Key::A;
            case GLFW_KEY_B: return Key::B;
            case GLFW_KEY_C: return Key::C;
            case GLFW_KEY_D: return Key::D;
            case GLFW_KEY_E: return Key::E;
            case GLFW_KEY_F: return Key::F;
            case GLFW_KEY_G: return Key::G;
            case GLFW_KEY_H: return Key::H;
            case GLFW_KEY_J: return Key::J;
            case GLFW_KEY_K: return Key::K;
            case GLFW_KEY_L: return Key::L;
            case GLFW_KEY_M: return Key::M;
            case GLFW_KEY_N: return Key::N;
            case GLFW_KEY_O: return Key::O;
            case GLFW_KEY_P: return Key::P;
            case GLFW_KEY_Q: return Key::Q;
            case GLFW_KEY_R: return Key::R;
            case GLFW_KEY_S: return Key::S;
            case GLFW_KEY_T: return Key::T;
            case GLFW_KEY_U: return Key::U;
            case GLFW_KEY_V: return Key::V;
            case GLFW_KEY_W: return Key::W;
            case GLFW_KEY_X: return Key::X;
            case GLFW_KEY_Y: return Key::Y;
            case GLFW_KEY_Z: return Key::Z;

            case GLFW_KEY_UP: return Key::UP;
            case GLFW_KEY_DOWN: return Key::DOWN;
            case GLFW_KEY_LEFT: return Key::LEFT;
            case GLFW_KEY_RIGHT: return Key::RIGHT;

            case GLFW_KEY_SPACE: return Key::SPACE;
            case GLFW_KEY_COMMA: return Key::COMMA;
            case GLFW_KEY_MINUS: return Key::MINUS;
            case GLFW_KEY_PERIOD: return Key::PERIOD;
            case GLFW_KEY_SLASH: return Key::SLASH;

            case GLFW_KEY_ENTER: return Key::ENTER;
            case GLFW_KEY_TAB: return Key::TAB;
            case GLFW_KEY_BACKSPACE: return Key::BACKSLASH;
            case GLFW_KEY_INSERT: return Key::INSERT;
            case GLFW_KEY_DELETE: return Key::DELETE;

            default: return Key::UNKNOWN;

        }
        // clang-format on
    }

    WindowSize getMonitorSize() {
        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);

        return {mode->width, mode->height};
    }

}// namespace

struct Canvas::Impl {

    Canvas& scope;
    GLFWwindow* window;

    WindowSize size_;
    Vector2 lastMousePos_;

    bool close_{false};

    std::priority_queue<Task, std::vector<Task>, CustomComparator> tasks_;
    std::optional<std::function<void(WindowSize)>> resizeListener;

    explicit Impl(Canvas& scope, const Canvas::Parameters& params)
        : scope(scope) {

        glfwSetErrorCallback(error_callback);

        if (!glfwInit()) {
            exit(EXIT_FAILURE);
        }

        if (params.size_) {
            size_ = *params.size_;
        } else {
            auto fullSize = getMonitorSize();
            size_ = {fullSize.width/2, fullSize.height/2};
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_RESIZABLE, params.resizable_);

        if (params.antialiasing_ > 0) {
            glfwWindowHint(GLFW_SAMPLES, params.antialiasing_);
        }

        window = glfwCreateWindow(size_.width, size_.height, params.title_.c_str(), nullptr, nullptr);
        if (!window) {
            glfwTerminate();
            exit(EXIT_FAILURE);
        }

        setWindowIcon(window, params.favicon_);

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

    bool animateOnce(const std::function<void()>& f) {

        if (close_ || glfwWindowShouldClose(window)) {
            return false;
        }

        handleTasks();

        f();

        glfwSwapBuffers(window);
        glfwPollEvents();

        return true;
    }

    void animate(const std::function<void()>& f) {

        while (animateOnce(f)) {}
    }

    void onWindowResize(std::function<void(WindowSize)> f) {
        this->resizeListener = std::move(f);
    }

    void invokeLater(const std::function<void()>& f, float t) {
        tasks_.emplace(f, static_cast<float>(glfwGetTime()) + t);
    }

    void close() {

        close_ = true;
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

        p->scope.onMouseWheelEvent({static_cast<float>(xoffset), static_cast<float>(yoffset)});
    }

    static void mouse_callback(GLFWwindow* w, int button, int action, int) {
        auto p = static_cast<Canvas::Impl*>(glfwGetWindowUserPointer(w));

        switch (action) {
            case GLFW_PRESS:
                p->scope.onMousePressedEvent(button, p->lastMousePos_, PeripheralsEventSource::MouseAction::PRESS);
                break;
            case GLFW_RELEASE:
                p->scope.onMousePressedEvent(button, p->lastMousePos_, PeripheralsEventSource::MouseAction::RELEASE);
                break;
            default:
                break;
        }
    }

    static void cursor_callback(GLFWwindow* w, double xpos, double ypos) {
        auto p = static_cast<Canvas::Impl*>(glfwGetWindowUserPointer(w));

        Vector2 mousePos(static_cast<float>(xpos), static_cast<float>(ypos));
        p->scope.onMouseMoveEvent(mousePos);
        p->lastMousePos_.copy(mousePos);
    }

    static void key_callback(GLFWwindow* w, int key, int scancode, int action, int mods) {

        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
            glfwSetWindowShouldClose(w, GLFW_TRUE);
            return;
        }

        KeyEvent evt{glfwKeyCodeToKey(key), scancode, mods};
        auto p = static_cast<Canvas::Impl*>(glfwGetWindowUserPointer(w));
        switch (action) {
            case GLFW_PRESS: {
                p->scope.onKeyEvent(evt, PeripheralsEventSource::KeyAction::PRESS);
                break;
            }
            case GLFW_RELEASE: {
                p->scope.onKeyEvent(evt, PeripheralsEventSource::KeyAction::RELEASE);
                break;
            }
            case GLFW_REPEAT: {
                p->scope.onKeyEvent(evt, PeripheralsEventSource::KeyAction::REPEAT);
                break;
            }
            default:
                break;
        }
    }
};

Canvas::Canvas(const Canvas::Parameters& params)
    : pimpl_(std::make_unique<Impl>(*this, params)) {}

Canvas::Canvas(const std::string& name)
    : Canvas(Canvas::Parameters().title(name)) {}

Canvas::Canvas(const std::string& name, const std::unordered_map<std::string, ParameterValue>& values)
    : Canvas(Canvas::Parameters(values).title(name)) {}


void Canvas::animate(const std::function<void()>& f) {

    pimpl_->animate(f);
}

bool Canvas::animateOnce(const std::function<void()>& f) {

    return pimpl_->animateOnce(f);
}

WindowSize Canvas::size() const {

    return pimpl_->getSize();
}

WindowSize Canvas::monitorSize() const {

    return getMonitorSize();
}

float Canvas::aspect() const {

    return size().aspect();
}

void Canvas::setSize(WindowSize size) {

    pimpl_->setSize(size);
}

void Canvas::onWindowResize(std::function<void(WindowSize)> f) {

    pimpl_->onWindowResize(std::move(f));
}

void Canvas::invokeLater(const std::function<void()>& f, float t) {

    pimpl_->invokeLater(f, t);
}

void Canvas::close() {

    pimpl_->close();
}

void* Canvas::windowPtr() const {

    return pimpl_->window;
}

Canvas::~Canvas() = default;


Canvas::Parameters::Parameters() = default;

Canvas::Parameters::Parameters(const std::unordered_map<std::string, ParameterValue>& values) {

    std::vector<std::string> unused;
    for (const auto& [key, value] : values) {

        bool used = false;

        if (key == "antialiasing" || key == "aa") {

            antialiasing(std::get<int>(value));
            used = true;

        } else if (key == "vsync") {

            vsync(std::get<bool>(value));
            used = true;

        } else if (key == "resizable") {

            resizable(std::get<bool>(value));
            used = true;

        } else if (key == "size") {

            auto _size = std::get<WindowSize>(value);
            size(_size);
            used = true;
        } else if (key == "favicon") {

            auto path = std::get<std::string>(value);
            favicon(path);
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

Canvas::Parameters& threepp::Canvas::Parameters::resizable(bool flag) {

    this->resizable_ = flag;

    return *this;
}

Canvas::Parameters& threepp::Canvas::Parameters::favicon(const std::filesystem::path& path) {

    if (std::filesystem::exists(path)) {
        favicon_ = path;
    } else {
        std::cerr << "Invalid favicon path: " << std::filesystem::absolute(path) << std::endl;
    }

    return *this;
}
