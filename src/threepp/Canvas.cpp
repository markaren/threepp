
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

    inline int sdlButtonConvert(Uint8 btn) {
        int result{};
        if (btn == SDL_BUTTON_LEFT) {
            result = 0;
        } else if (btn == SDL_BUTTON_RIGHT) {
            result = 1;
        } else if (btn == SDL_BUTTON_MIDDLE) {
            result = 2;
        }

        return result;
    }

    inline float time() {
        return static_cast<float>(SDL_GetTicks()) / 1000;
    }

    inline Key getKey(SDL_Keycode key) {

        switch (key) {
            case SDLK_0: return Key::NUM_0;
            case SDLK_1: return Key::NUM_1;
            case SDLK_2: return Key::NUM_2;
            case SDLK_3: return Key::NUM_3;
            case SDLK_4: return Key::NUM_4;
            case SDLK_5: return Key::NUM_5;
            case SDLK_6: return Key::NUM_6;
            case SDLK_7: return Key::NUM_7;
            case SDLK_8: return Key::NUM_8;
            case SDLK_9: return Key::NUM_9;

            case SDLK_a: return Key::A;
            case SDLK_b: return Key::B;
            case SDLK_c: return Key::C;
            case SDLK_d: return Key::D;
            case SDLK_e: return Key::E;
            case SDLK_f: return Key::F;
            case SDLK_g: return Key::G;
            case SDLK_h: return Key::H;
            case SDLK_j: return Key::J;
            case SDLK_k: return Key::K;
            case SDLK_l: return Key::L;
            case SDLK_m: return Key::M;
            case SDLK_n: return Key::N;
            case SDLK_o: return Key::O;
            case SDLK_p: return Key::P;
            case SDLK_r: return Key::R;
            case SDLK_s: return Key::S;
            case SDLK_t: return Key::T;
            case SDLK_u: return Key::U;
            case SDLK_v: return Key::V;
            case SDLK_w: return Key::W;
            case SDLK_x: return Key::X;
            case SDLK_y: return Key::Y;
            case SDLK_z: return Key::Z;

            case SDLK_UP: return Key::UP;
            case SDLK_DOWN: return Key::DOWN;
            case SDLK_LEFT: return Key::LEFT;
            case SDLK_RIGHT: return Key::RIGHT;

            case SDLK_SPACE: return Key::SPACE;

            default: return Key::UNKNOWN;
        }
    }

}// namespace

struct Canvas::Impl {

    SDL_Window* window;
    SDL_Renderer* renderer;
    SDL_GLContext maincontext;

    IOCapture* ioCapture;

    WindowSize size_;

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

        gladLoadGL();

        SDL_GL_SetSwapInterval(params.vsync_ ? 1 : 0);

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


    void handleEvents() {
        if (event.type == SDL_WINDOWEVENT) {
            if (event.window.event == SDL_WINDOWEVENT_RESIZED && resizeListener) {
                WindowSize s{};
                SDL_GetWindowSize(window, &s.width, &s.height);
                resizeListener->operator()(s);
            }
        } else if (event.type == SDL_MOUSEBUTTONDOWN) {

            if (ioCapture && (ioCapture->preventMouseEvent)()) {
                return;
            }

            auto listeners = mouseListeners;
            for (auto l : listeners) {
                auto action = event.button;
                l->onMouseDown(sdlButtonConvert(action.button), {action.x, action.y});
            }

        } else if (event.type == SDL_MOUSEBUTTONUP) {

            if (ioCapture && (ioCapture->preventMouseEvent)()) {
                return;
            }

            for (auto l : mouseListeners) {
                auto action = event.button;
                l->onMouseUp(sdlButtonConvert(action.button), {action.x, action.y});
            }

        } else if (event.type == SDL_MOUSEMOTION) {

            if (ioCapture && (ioCapture->preventMouseEvent)()) {
                return;
            }

            auto listeners = mouseListeners;
            for (auto l : listeners) {
                auto action = event.button;
                l->onMouseMove({action.x, action.y});
            }

        } else if (event.type == SDL_MOUSEWHEEL) {

            if (ioCapture && (ioCapture->preventScrollEvent)()) {
                return;
            }

            auto listeners = mouseListeners;
            for (auto l : listeners) {
                auto action = event.wheel;
                l->onMouseWheel({action.x, action.y});
            }

        } else if (event.type == SDL_KEYDOWN) {

            if (ioCapture && (ioCapture->preventKeyboardEvent)()) {
                return;
            }

            KeyEvent evt{getKey(event.key.keysym.sym), event.key.keysym.scancode, event.key.keysym.mod};
            auto listeners = keyListeners;
            for (auto l : listeners) {

                if (event.key.repeat) {
                    l->onKeyRepeat(evt);
                } else {
                    l->onKeyPressed(evt);
                }
            }

        } else if (event.type == SDL_KEYUP) {

            if (ioCapture && (ioCapture->preventKeyboardEvent)()) {
                return;
            }

            KeyEvent evt{getKey(event.key.keysym.sym), event.key.keysym.scancode, event.key.keysym.mod};
            auto listeners = keyListeners;
            for (auto l : listeners) {

                l->onKeyReleased(evt);
            }
        }
    }

    ~Impl() {
        SDL_GL_DeleteContext(maincontext);
        SDL_DestroyWindow(window);
        SDL_Quit();
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
