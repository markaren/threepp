
#include "threepp/Canvas.hpp"
#include "threepp/loaders/ImageLoader.hpp"

#include "threepp/utils/StringUtils.hpp"

#include "threepp/favicon.hpp"

#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
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

    void setWindowIcon(SDL_Window* window, std::optional<std::filesystem::path> customIcon) {

        ImageLoader imageLoader;
        std::optional<Image> favicon;
        if (customIcon) {
            favicon = imageLoader.load(*customIcon, Image::Format::RGBA, false);
        } else {
            favicon = imageLoader.load(faviconSource(), Image::Format::RGBA, false);
        }
        if (favicon) {
            Uint32 rmask, gmask, bmask, amask;
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
            rmask = 0xff000000;
            gmask = 0x00ff0000;
            bmask = 0x0000ff00;
            amask = 0x000000ff;
#else
            rmask = 0x000000ff;
            gmask = 0x0000ff00;
            bmask = 0x00ff0000;
            amask = 0xff000000;
#endif
            int numChannels = favicon->numChannels();
            SDL_Surface* s = SDL_CreateRGBSurfaceFrom(favicon->getData(), static_cast<int>(favicon->width), static_cast<int>(favicon->height), numChannels * 8, numChannels * favicon->width, rmask, gmask, bmask, amask);
            SDL_SetWindowIcon(window, s);
            SDL_FreeSurface(s);
        }
    }

    /** A simple function that prints a message, the error code returned by SDL,
     * and quits the application
     */
    void sdldie(const std::string& msg) {
        std::cerr << msg << ": " << SDL_GetError() << std::endl;
        SDL_Quit();
        exit(1);
    }

    inline float time() {
        return static_cast<float>(SDL_GetTicks()) / 1000;
    }

}// namespace

struct Canvas::Impl {

    SDL_Window* window;
    SDL_Renderer* renderer;
    SDL_GLContext maincontext;

    IOCapture* ioCapture;

    WindowSize size_;
    Vector2 lastMousePos_;

    SDL_Event event;

    std::priority_queue<task, std::vector<task>, CustomComparator> tasks_;
    std::optional<std::function<void(WindowSize)>> resizeListener;
    std::vector<KeyListener*> keyListeners;
    std::vector<MouseListener*> mouseListeners;


    explicit Impl(const Canvas::Parameters& params)
        : size_(params.size_), ioCapture(nullptr) {

        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            sdldie("");
        }
        SDL_GL_LoadLibrary(nullptr);// Default OpenGL is fine.

        //        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        //        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        //        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
        //        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        //        glfwWindowHint(GLFW_RESIZABLE, params.resizable_);

        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);

        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

        if (params.antialiasing_ > 0) {
            SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
            SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, params.antialiasing_);
        }

        Uint32 flags = SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN;
        if (params.resizable_) {
            flags = flags | SDL_WINDOW_RESIZABLE;
        }

        window = SDL_CreateWindow(params.title_.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                  params.size_.width, params.size_.height, flags);
        if (!window) {
            sdldie("");
        }

        maincontext = SDL_GL_CreateContext(window);

        setWindowIcon(window, params.favicon_);

        //        glfwSetWindowUserPointer(window, this);

        //        glfwSetKeyCallback(window, key_callback);
        //        glfwSetMouseButtonCallback(window, mouse_callback);
        //        glfwSetCursorPosCallback(window, cursor_callback);
        //        glfwSetScrollCallback(window, scroll_callback);
        //        glfwSetWindowSizeCallback(window, window_size_callback);

        //        glfwMakeContextCurrent(window);
        gladLoadGL();
        //        glfwSwapInterval(params.vsync_ ? 1 : 0);

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
        SDL_SetWindowSize(window, size.width, size.height);
    }

    inline void handleTasks() {
        while (!tasks_.empty()) {
            auto& task = tasks_.top();
            if (task.second < time()) {
                task.first();
                tasks_.pop();
            } else {
                break;
            }
        }
    }

    void handleEvents() {
        if (event.type == SDL_WINDOWEVENT) {
            if (event.window.event == SDL_WINDOWEVENT_RESIZED && resizeListener) {
                WindowSize s{};
                SDL_GetWindowSize(window, &s.width, &s.height);
                resizeListener->operator()(s);
            }
        } else if (event.type == SDL_MOUSEBUTTONDOWN) {

            for (auto l : mouseListeners) {
                auto action = event.button;
                int btn{};
                switch (action.button) {
                    case SDL_BUTTON_LEFT: {
                        btn = 0;
                        break;
                    }
                    case SDL_BUTTON_RIGHT: {
                        btn = 1;
                        break;
                    }
                    case SDL_BUTTON_MIDDLE: {
                        btn = 2;
                        break;
                    }
                }
                l->onMouseDown(btn, {action.x, action.y});
            }

        } else if (event.type == SDL_MOUSEBUTTONUP) {

            for (auto l : mouseListeners) {
                auto action = event.button;
                int btn{};
                switch (action.button) {
                    case SDL_BUTTON_LEFT: {
                        btn = 0;
                    }
                    case SDL_BUTTON_RIGHT: {
                        btn = 1;
                    }
                    case SDL_BUTTON_MIDDLE: {
                        btn = 2;
                    }
                }

                l->onMouseUp(btn, {action.x, action.y});
            }

        } else if (event.type == SDL_MOUSEMOTION) {

            for (auto l : mouseListeners) {
                auto action = event.button;
                l->onMouseMove({action.x, action.y});
            }

        } else if (event.type == SDL_MOUSEWHEEL) {

            for (auto l : mouseListeners) {
                auto action = event.wheel;
                l->onMouseWheel({action.x, action.y});
            }

        }
    }

    bool animateOnce(const std::function<void()>& f) {

        SDL_GL_SwapWindow(window);

        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                return false;
            } else {
                handleEvents();
            }
        }

        handleTasks();

        f();

        return true;
    }

    void animate(const std::function<void()>& f) {

        while (animateOnce(f)) {}
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
        tasks_.emplace(f, static_cast<float>(SDL_GetTicks()) + t);
    }

    ~Impl() {
        SDL_GL_DeleteContext(maincontext);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }

    static void window_size_callback(SDL_Window* w, int width, int height) {
        //        auto p = static_cast<Canvas::Impl*>(glfwGetWindowUserPointer(w));
        //        p->size_ = {width, height};
        //        if (p->resizeListener) p->resizeListener.value().operator()(p->size_);
    }

    static void error_callback(int error, const char* description) {
        std::cerr << "Error: " << description << std::endl;
    }

    static void scroll_callback(SDL_Window* w, double xoffset, double yoffset) {
        //        auto p = static_cast<Canvas::Impl*>(glfwGetWindowUserPointer(w));
        //
        //        if (p->ioCapture && (p->ioCapture->preventScrollEvent)()) {
        //            return;
        //        }
        //
        //        auto listeners = p->mouseListeners;
        //        if (listeners.empty()) return;
        //        Vector2 delta{(float) xoffset, (float) yoffset};
        //        for (auto l : listeners) {
        //            l->onMouseWheel(delta);
        //        }
    }

    static void mouse_callback(SDL_Window* w, int button, int action, int mods) {
        //        auto p = static_cast<Canvas::Impl*>(glfwGetWindowUserPointer(w));
        //
        //        if (p->ioCapture && (p->ioCapture->preventMouseEvent)()) {
        //            return;
        //        }
        //
        //        auto listeners = p->mouseListeners;
        //        for (auto l : listeners) {
        //
        //            switch (action) {
        //                case GLFW_PRESS:
        //                    l->onMouseDown(button, p->lastMousePos_);
        //                    break;
        //                case GLFW_RELEASE:
        //                    l->onMouseUp(button, p->lastMousePos_);
        //                    break;
        //                default:
        //                    break;
        //            }
        //        }
    }

    static void cursor_callback(SDL_Window* w, double xpos, double ypos) {
        //        auto p = static_cast<Canvas::Impl*>(glfwGetWindowUserPointer(w));
        //
        //        if (p->ioCapture && (p->ioCapture->preventMouseEvent)()) {
        //            return;
        //        }
        //
        //        p->lastMousePos_.set(static_cast<float>(xpos), static_cast<float>(ypos));
        //        auto listeners = p->mouseListeners;
        //        for (auto l : listeners) {
        //            l->onMouseMove(p->lastMousePos_);
        //        }
    }

    static void key_callback(SDL_Window* w, int key, int scancode, int action, int mods) {
        //        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        //            glfwSetWindowShouldClose(w, GLFW_TRUE);
        //            return;
        //        }
        //
        //        auto p = static_cast<Canvas::Impl*>(glfwGetWindowUserPointer(w));
        //
        //        if (p->ioCapture && (p->ioCapture->preventKeyboardEvent)()) {
        //            return;
        //        }
        //
        //        if (p->keyListeners.empty()) return;
        //
        //        KeyEvent evt{key, scancode, mods};
        //        auto listeners = p->keyListeners;
        //        for (auto l : listeners) {
        //            switch (action) {
        //                case GLFW_PRESS:
        //                    l->onKeyPressed(evt);
        //                    break;
        //                case GLFW_RELEASE:
        //                    l->onKeyReleased(evt);
        //                    break;
        //                case GLFW_REPEAT:
        //                    l->onKeyRepeat(evt);
        //                    break;
        //                default:
        //                    break;
        //            }
        //        }
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

bool Canvas::animateOnce(const std::function<void()>& f) {

    return pimpl_->animateOnce(f);
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

void Canvas::setIOCapture(IOCapture* capture) {

    pimpl_->ioCapture = capture;
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

void* Canvas::windowPtr() const {

    return pimpl_->window;
}

Canvas::~Canvas() = default;


Canvas::Parameters::Parameters() = default;

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
