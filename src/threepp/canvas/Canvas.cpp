
#include "threepp/canvas/Canvas.hpp"
#include "threepp/canvas/Monitor.hpp"

#include "threepp/favicon.hpp"
#include "threepp/loaders/ImageLoader.hpp"
#include "threepp/utils/StringUtils.hpp"

#ifndef __EMSCRIPTEN__
#include "threepp/utils/LoadGlad.hpp"
#define GLFW_INCLUDE_NONE
#else
#include <emscripten.h>
#endif

#include <GLFW/glfw3.h>

#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string_view>

using namespace threepp;

namespace {

#ifdef __EMSCRIPTEN__
    struct FunctionWrapper {
        std::function<void()> loopFunction;
        std::function<void()> frameEndCallback;

        FunctionWrapper(std::function<void()> loopFunction, std::function<void()> frameEndCallback)
            : loopFunction(std::move(loopFunction)), frameEndCallback(std::move(frameEndCallback)) {}

        void loop() {
            loopFunction();
            if (frameEndCallback) {
                frameEndCallback();
            }
        }
    };

    EMSCRIPTEN_KEEPALIVE
    void emscriptenLoop(void* arg) {
        static_cast<FunctionWrapper*>(arg)->loop();
    }

#else
    void setWindowIcon(GLFWwindow* window, std::optional<std::filesystem::path> customIcon) {

#ifdef __APPLE__
        return;// operation is not supported on macOS
#endif

        ImageLoader imageLoader;
        std::optional<Image> favicon;
        if (customIcon) {
            favicon = imageLoader.load(*customIcon, 4, false);
        } else {
            favicon = imageLoader.load(faviconSource(), 4, false);
        }
        if (favicon) {
            GLFWimage images[1];
            images[0] = {static_cast<int>(favicon->width()),
                         static_cast<int>(favicon->height()),
                         favicon->data().data()};
            glfwSetWindowIcon(window, 1, images);
        }
    }
#endif

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
            case GLFW_KEY_BACKSPACE: return Key::BACKSPACE;
            case GLFW_KEY_INSERT: return Key::INSERT;
            case GLFW_KEY_DELETE: return Key::DEL;

            case GLFW_KEY_LEFT_SHIFT: return Key::LEFT_SHIFT;
            case GLFW_KEY_LEFT_CONTROL: return Key::LEFT_CONTROL;
            case GLFW_KEY_LEFT_ALT: return Key::LEFT_ALT;

            default: return Key::UNKNOWN;

        }
        // clang-format on
    }
    void error_callback(int /*error*/, const char* description) {
        std::cerr << "Error: " << description << std::endl;
    }

    static int& glfwRefCount() {
        static int count = 0;
        return count;
    }

    void initGLfw(bool headless) {
        if (glfwRefCount()++ == 0) {
            glfwSetErrorCallback(error_callback);
            (void) headless;
#if !defined(__EMSCRIPTEN__) && defined(GLFW_PLATFORM)
            // Pick the GLFW platform before the first glfwInit reads the hint.
            // A headless canvas on a machine with no display server (cloud GPU
            // instances: DISPLAY/WAYLAND_DISPLAY unset) selects the Null
            // platform — glfwInit would otherwise fail outright on X11/Wayland,
            // and a headless Vulkan canvas never needs the window system: its
            // surface comes from VK_EXT_headless_surface (see VulkanContext).
            // Windows and macOS always have a window system, so headless keeps
            // the native platform (hidden window). THREEPP_GLFW_PLATFORM=null
            // forces the Null platform anywhere, which is how the display-free
            // path is exercised on a developer machine. The hint is sticky
            // across init/terminate cycles, so the windowed case must reset it
            // to ANY_PLATFORM.
            bool wantNull = false;
            if (const char* forced = std::getenv("THREEPP_GLFW_PLATFORM"); forced && *forced) {
                wantNull = std::string_view{forced} == "null";
            }
#if !defined(_WIN32) && !defined(__APPLE__)
            else if (headless) {
                const char* x11 = std::getenv("DISPLAY");
                const char* wl = std::getenv("WAYLAND_DISPLAY");
                wantNull = (!x11 || !*x11) && (!wl || !*wl);
            }
#endif
            glfwInitHint(GLFW_PLATFORM, wantNull ? GLFW_PLATFORM_NULL : GLFW_ANY_PLATFORM);
#elif !defined(__EMSCRIPTEN__)
            // GLFW before 3.4 has no platform selection and no Null platform:
            // a display-less machine just fails in glfwInit below. Distro
            // packages still ship 3.3 (Ubuntu 22.04), so an external GLFW can
            // land here even though the vendored copy is 3.4.
            if (const char* forced = std::getenv("THREEPP_GLFW_PLATFORM"); forced && std::string_view{forced} == "null") {
                std::cerr << "Canvas: THREEPP_GLFW_PLATFORM=null ignored - GLFW "
                          << GLFW_VERSION_MAJOR << '.' << GLFW_VERSION_MINOR
                          << " predates the Null platform (needs 3.4)" << std::endl;
            }
#endif
            if (!glfwInit()) {
                --glfwRefCount();
                throw std::runtime_error("Canvas: glfwInit() failed");
            }
        }
    }

    void termGLfw() {
        if (--glfwRefCount() == 0) {
            glfwTerminate();
        }
    }

    // The primary monitor's current video mode, in screen coordinates — the
    // size a borderless-fullscreen window takes. Unlike monitor::monitorSize()
    // this takes no GLFW reference of its own (it assumes glfwInit has already
    // run, which it has by the time the Canvas ctor asks), so a fullscreen
    // canvas still terminates GLFW when it is destroyed. Returns {0,0} if
    // there is no monitor to ask, which the caller treats as "not fullscreen".
    WindowSize primaryMonitorSize() {
#ifdef __EMSCRIPTEN__
        return monitor::monitorSize();
#else
        if (GLFWmonitor* m = glfwGetPrimaryMonitor()) {
            if (const GLFWvidmode* mode = glfwGetVideoMode(m)) {
                return {mode->width, mode->height};
            }
        }
        return {0, 0};
#endif
    }

}// namespace

struct Canvas::Impl {

    Canvas& scope;
    GLFWwindow* window{nullptr};

    WindowSize size_;
    Vector2 lastMousePos_;

    bool close_{false};

    // Retained for lazy window creation (initWindow called on first animate).
    Parameters params_;

    std::vector<std::function<void(WindowSize)>> resizeListener;
    std::vector<std::function<void(int monitor)>> monitorChangesListener;
    std::function<void()> frameEndCallback_;
    bool insideAnimateLoop_{false};

    explicit Impl(Canvas& scope, const Parameters& params)
        : scope(scope),
          params_(params) {

        initGLfw(params.headless_);

        // Fullscreen takes the monitor's size and ignores any requested one.
        // Headless wins over it: there is no window to make borderless, and a
        // display-less machine has no monitor to measure.
        const WindowSize screen = borderlessFullscreen() ? primaryMonitorSize() : WindowSize{0, 0};
        if (screen.width() > 0 && screen.height() > 0) {
            size_ = screen;
        } else if (params.size_) {
            size_ = *params.size_;
        } else {
            const auto fullSize = monitor::monitorSize();
            size_ = {fullSize.width() / 2, fullSize.height() / 2};
        }
    }

    // Borderless windowed fullscreen requested AND applicable. Headless has no
    // window at all, so it wins; see Parameters::fullscreen.
    [[nodiscard]] bool borderlessFullscreen() const {
        return params_.fullscreen_ && !params_.headless_;
    }

    void initWindow(GraphicsAPI api) {
        if (window) return; // already initialised
        params_.graphicsApi_ = api;

#ifndef __EMSCRIPTEN__
        // GLFW window hints are sticky, process-global state: whatever the last
        // canvas asked for still applies to the next glfwCreateWindow. A Vulkan
        // canvas sets GLFW_CLIENT_API=GLFW_NO_API, so without this reset a GL
        // canvas created afterwards silently got a window with no GL context
        // (and the same leak applied to GLFW_SAMPLES). Start from a known state.
        glfwDefaultWindowHints();

        if (api == GraphicsAPI::Vulkan) {
            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        } else {
            glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
            glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        }
        // Borderless windowed fullscreen: strip the decorations and pin the
        // size. Deliberately NOT glfwCreateWindow(..., monitor, ...) — an
        // exclusive-fullscreen window owns the display mode, cannot stay hidden
        // until the first present (see deferShow below), and mode-switches the
        // monitor on every alt-tab. An undecorated window covering the monitor
        // looks the same and keeps all of that machinery intact.
        const bool borderless = borderlessFullscreen();
        if (borderless) {
            glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
        }
        glfwWindowHint(GLFW_RESIZABLE, borderless ? GLFW_FALSE : params_.resizable_);
        // A Vulkan canvas starts hidden and is revealed by the renderer once it
        // has presented a first frame (Canvas::showWindow). Device and pipeline
        // setup leaves a visible window blank and unresponsive for seconds —
        // long enough that the user clicks back to whatever launched the app,
        // and the first real frame then surfaces in the background. GL windows
        // paint within the same frame they appear, so they stay visible-on-create.
        // A borderless-fullscreen GL window is also created hidden, so that it
        // can be moved onto the monitor origin before it is ever painted —
        // otherwise it flashes at GLFW's default (centred) position first. It
        // is revealed at the end of this function, unlike a Vulkan canvas,
        // which stays hidden until its renderer has presented.
        const bool deferShow = params_.headless_ || api == GraphicsAPI::Vulkan;
        glfwWindowHint(GLFW_VISIBLE, (deferShow || borderless) ? GLFW_FALSE : GLFW_TRUE);
#else
        // Browser: OpenGL (WebGL2) needs GLFW to create the WebGL context.
        // Suppressing it left GLctx undefined and crashed the GL renderer on
        // startup.
        glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#endif

        if (params_.antialiasing_ > 0) {
            glfwWindowHint(GLFW_SAMPLES, params_.antialiasing_);
        }

        window = glfwCreateWindow(size_.width(), size_.height(), params_.title_.c_str(), nullptr, nullptr);
        if (!window) {
            termGLfw();
            throw std::runtime_error(
                    "Canvas: glfwCreateWindow failed for '" + params_.title_ + "' (requested " +
                    (api == GraphicsAPI::Vulkan ? "Vulkan" : "OpenGL") + ")");
        }

        // NOTE, because it is the root of a crash that took a while to find:
        // size_ is what was REQUESTED and the platform is free to disagree.
        // Windows clamps a window's client area to the desktop work area (a
        // 1200-tall window on a 1200-tall monitor comes back 1181 tall with a
        // taskbar on screen) and enforces a minimum width (a 64-wide window is
        // really 120 wide). GLFW fires no resize callback for a size it only
        // ever set once, at creation, so size_ keeps describing a window that
        // never existed. Syncing it from glfwGetWindowSize here is NOT safe:
        // size_ also drives the GL viewport and the camera aspect, and the
        // small offscreen-style canvases the GL tests render into would start
        // rendering at the platform's minimum width instead of the size they
        // asked for. Consumers that pair this number with a driver-sized
        // buffer must therefore ask the driver, not the canvas — see
        // VulkanRenderer::writeFramebuffer and Impl::Impl, which both size
        // from the swapchain extent for exactly this reason.

#ifndef __EMSCRIPTEN__
        if (borderless) {
            // Cover the primary monitor from its own origin (which is NOT
            // (0,0) on a multi-monitor desktop whose primary sits to the right
            // of another screen).
            int mx = 0, my = 0;
            if (GLFWmonitor* m = glfwGetPrimaryMonitor()) {
                glfwGetMonitorPos(m, &mx, &my);
            }
            glfwSetWindowPos(window, mx, my);
            // The reveal waits until the icon and callbacks are installed
            // below — see the end of this function.
        }
#endif

#ifdef __EMSCRIPTEN__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdollar-in-identifier-extension"
        EM_ASM({ document.title = UTF8ToString($0); }, params_.title_.c_str());
#pragma GCC diagnostic pop
#endif

        glfwSetWindowUserPointer(window, this);

#ifndef __EMSCRIPTEN__
        setWindowIcon(window, params_.favicon_);
#endif

        glfwSetKeyCallback(window, key_callback);
        glfwSetMouseButtonCallback(window, mouse_callback);
        glfwSetCursorPosCallback(window, cursor_callback);
        glfwSetScrollCallback(window, scroll_callback);
        glfwSetWindowSizeCallback(window, window_size_callback);
        glfwSetWindowPosCallback(window, window_pos_callback);
        glfwSetDropCallback(window, drop_callback);

        if (api == GraphicsAPI::OpenGL) {
#ifndef __EMSCRIPTEN__
            // Belt and braces for the hint reset above: a context-less window
            // would otherwise fail deep inside glad with no useful message.
            if (glfwGetWindowAttrib(window, GLFW_CLIENT_API) == GLFW_NO_API) {
                throw std::runtime_error(
                        "Canvas: OpenGL was requested for '" + params_.title_ +
                        "' but the window was created without a GL context "
                        "(GLFW_CLIENT_API=GLFW_NO_API leaked from a Vulkan canvas)");
            }
#endif
            glfwMakeContextCurrent(window);

#ifndef __EMSCRIPTEN__
            loadGlad();
            glfwSwapInterval(params_.vsync_ ? 1 : 0);

            if (params_.antialiasing_ > 0) {
                glEnable(GL_MULTISAMPLE);
            }

            glEnable(GL_PROGRAM_POINT_SIZE);
#endif
        }

#ifndef __EMSCRIPTEN__
        // A borderless-fullscreen window was created hidden only so it could be
        // moved onto the monitor origin unseen; now that it is placed, iconed
        // and wired up, reveal it. A canvas with a real reason to stay hidden
        // (headless, or Vulkan waiting on its first present) is left alone.
        if (borderless && !deferShow) {
            glfwShowWindow(window);
        }
#endif
    }

    [[nodiscard]] const WindowSize& getSize() const {

        return size_;
    }

    void setSize(std::pair<int, int> size) {

        if (!window) {
            params_.size(size);
            return;
        }

        glfwSetWindowSize(window, size.first, size.second);
    }

    bool animateOnce(const std::function<void()>& f) {

        if (!window) initWindow(params_.graphicsApi_);

        if (close_ || glfwWindowShouldClose(window)) {
            close_ = true;
            return false;
        }

        insideAnimateLoop_ = true;
        f();
        insideAnimateLoop_ = false;

        if (params_.graphicsApi_ == GraphicsAPI::OpenGL) {
            glfwSwapBuffers(window);
        } else if (frameEndCallback_) {
            frameEndCallback_();
        }
        glfwPollEvents();

        return true;
    }

    void animate(const std::function<void()>& f) {
#ifdef __EMSCRIPTEN__
        FunctionWrapper wrapper(f, frameEndCallback_);
        emscripten_set_main_loop_arg(&emscriptenLoop, &wrapper, 0, true);
#else
        while (animateOnce(f)) {}
#endif
    }

    void onWindowResize(std::function<void(WindowSize)> f) {
        this->resizeListener.emplace_back(std::move(f));
    }

    void onMonitorChange(std::function<void(int)> f) {
        this->monitorChangesListener.emplace_back(std::move(f));
    }

    void close() {

        close_ = true;
    }

    ~Impl() {
        if (window) {
            // Hand the pointer back BEFORE the window goes away, for apps that
            // grabbed it for mouse-look (GLFW_CURSOR_DISABLED + raw motion —
            // see the FPS/TPS demos). The bundled GLFW does call enableCursor()
            // from its destroy path, but GLFW 3.3.x does not (it only nulls
            // _glfw.win32.disabledCursorWindow), so a system/older GLFW leaves
            // the OS cursor hidden, clipped and in raw-motion mode after the
            // process ends. Doing it here makes the release explicit and
            // version-independent. Paths that never reach this destructor at
            // all (std::exit, Ctrl+C, a crash) still have to release it
            // themselves — the demos do so before their std::exit.
            if (glfwRawMouseMotionSupported())// else GLFW_FEATURE_UNAVAILABLE
                glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_FALSE);
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            glfwDestroyWindow(window);
        }
        termGLfw();
    }


    static void window_pos_callback(GLFWwindow* w, int wx, int wy) {

        auto p = static_cast<Impl*>(glfwGetWindowUserPointer(w));

        int count;
        GLFWmonitor** monitors = glfwGetMonitors(&count);

        // For each monitor, get its bounds
        for (int i = 0; i < count; ++i) {
            int mx, my;
            glfwGetMonitorPos(monitors[i], &mx, &my);
            const GLFWvidmode* mode = glfwGetVideoMode(monitors[i]);
            const int mw = mode->width;
            const int mh = mode->height;

            // Check if window is within this monitor's bounds
            if (wx >= mx && wx < mx + mw && wy >= my && wy < my + mh) {
                for (const auto& listener : p->monitorChangesListener) {
                    listener(i);
                }
                break;
            }
        }
    }


    static void window_size_callback(GLFWwindow* w, int width, int height) {
        auto p = static_cast<Impl*>(glfwGetWindowUserPointer(w));
        p->size_ = {width, height};
        for (const auto& listener : p->resizeListener) {
            listener(p->size_);
        }
    }


    static void scroll_callback(GLFWwindow* w, double xoffset, double yoffset) {
        const auto p = static_cast<Impl*>(glfwGetWindowUserPointer(w));

        p->scope.onMouseWheelEvent({static_cast<float>(xoffset), static_cast<float>(yoffset)});
    }

    static void mouse_callback(GLFWwindow* w, int button, int action, int) {
        const auto p = static_cast<Impl*>(glfwGetWindowUserPointer(w));

        switch (action) {
            case GLFW_PRESS:
                p->scope.onMousePressedEvent(button, p->lastMousePos_, MouseAction::PRESS);
                break;
            case GLFW_RELEASE:
                p->scope.onMousePressedEvent(button, p->lastMousePos_, MouseAction::RELEASE);
                break;
            default:
                break;
        }
    }

    static void cursor_callback(GLFWwindow* w, double xpos, double ypos) {
        const auto p = static_cast<Impl*>(glfwGetWindowUserPointer(w));

        Vector2 mousePos(static_cast<float>(xpos), static_cast<float>(ypos));
        p->scope.onMouseMoveEvent(mousePos);
        p->lastMousePos_.copy(mousePos);
    }

    static void key_callback(GLFWwindow* w, int key, int scancode, int action, int mods) {

        const auto p = static_cast<Impl*>(glfwGetWindowUserPointer(w));

        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS && p->params_.exitOnKeyEscape_) {
            glfwSetWindowShouldClose(w, GLFW_TRUE);
            return;
        }

        const KeyEvent evt{glfwKeyCodeToKey(key), scancode, mods};
        switch (action) {
            case GLFW_PRESS: {
                p->scope.onKeyEvent(evt, KeyAction::PRESS);
                break;
            }
            case GLFW_RELEASE: {
                p->scope.onKeyEvent(evt, KeyAction::RELEASE);
                break;
            }
            case GLFW_REPEAT: {
                p->scope.onKeyEvent(evt, KeyAction::REPEAT);
                break;
            }
            default:
                break;
        }
    }

    static void drop_callback(GLFWwindow* w, int count, const char** paths) {

        auto p = static_cast<Impl*>(glfwGetWindowUserPointer(w));

        std::vector<std::string> v;
        for (int i = 0; i < count; ++i) {
            v.emplace_back(paths[i]);
        }

        p->scope.onDropEvent(v);
    }
};

Canvas::Canvas(const Parameters& params)
    : pimpl_(std::make_unique<Impl>(*this, params)) {}

Canvas::Canvas(const std::string& name)
    : Canvas(Parameters().title(name)) {}

Canvas::Canvas(const std::string& name, const std::unordered_map<std::string, ParameterValue>& values)
    : Canvas(Parameters(values).title(name)) {}


void Canvas::animate(const std::function<void()>& f) {

    pimpl_->animate(f);
}

bool Canvas::animateOnce(const std::function<void()>& f) {

    return pimpl_->animateOnce(f);
}

bool Canvas::isOpen() const {

    return !pimpl_->close_;
}

WindowSize Canvas::size() const {

    return pimpl_->getSize();
}

float Canvas::aspect() const {

    return size().aspect();
}

void Canvas::exitOnKeyEscape(bool value) {
    pimpl_->params_.exitOnKeyEscape_ = value;
}

void Canvas::setSize(std::pair<int, int> size) {

    pimpl_->setSize(size);
}

void Canvas::onWindowResize(std::function<void(WindowSize)> f) {

    pimpl_->onWindowResize(std::move(f));
}

void Canvas::onMonitorChange(std::function<void(int)> f) const {

    pimpl_->onMonitorChange(std::move(f));
}

void Canvas::close() {

    pimpl_->close();
}

void* Canvas::windowPtr() const {

    return pimpl_->window;
}

GraphicsAPI Canvas::graphicsApi() const {

    return pimpl_->params_.graphicsApi_;
}

void Canvas::initWindow(GraphicsAPI api) {

    pimpl_->initWindow(api);
}

bool Canvas::vsync() const {

    return pimpl_->params_.vsync_;
}

int Canvas::samples() const {

    return pimpl_->params_.antialiasing_;
}

bool Canvas::headless() const {

    return pimpl_->params_.headless_;
}

void Canvas::setFrameEndCallback(std::function<void()> callback) {
    pimpl_->frameEndCallback_ = std::move(callback);
}

void Canvas::showWindow() {

#ifndef __EMSCRIPTEN__
    // Never un-hide a headless canvas: hidden IS its contract, and on the GLFW
    // Null platform there is nothing to show anyway.
    if (pimpl_->params_.headless_) return;
    if (!pimpl_->window) return;
    if (glfwGetWindowAttrib(pimpl_->window, GLFW_VISIBLE)) return;
    glfwShowWindow(pimpl_->window);
#endif
}

bool Canvas::isInsideAnimateLoop() const {
    return pimpl_->insideAnimateLoop_;
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

            size(std::get<WindowSize>(value));
            used = true;

        } else if (key == "favicon") {

            auto path = std::get<std::string>(value);
            favicon(path);
            used = true;

        } else if (key == "exitOnKeyEscape") {

            exitOnKeyEscape(std::get<bool>(value));
            used = true;

        } else if (key == "headless") {

            headless(std::get<bool>(value));
            used = true;

        } else if (key == "fullscreen") {

            fullscreen(std::get<bool>(value));
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

Canvas::Parameters& Canvas::Parameters::resizable(bool flag) {

    this->resizable_ = flag;

    return *this;
}

Canvas::Parameters& Canvas::Parameters::favicon(const std::filesystem::path& path) {

    if (exists(path)) {
        favicon_ = path;
    } else {
        std::cerr << "Invalid favicon path: " << absolute(path) << std::endl;
    }

    return *this;
}

Canvas::Parameters& Canvas::Parameters::exitOnKeyEscape(bool flag) {

    exitOnKeyEscape_ = flag;

    return *this;
}

Canvas::Parameters& Canvas::Parameters::headless(bool flag) {

    headless_ = flag;

    return *this;
}

Canvas::Parameters& Canvas::Parameters::fullscreen(bool flag) {

    fullscreen_ = flag;

    return *this;
}

WindowSize monitor::monitorSize(int monitor) {

#ifdef __EMSCRIPTEN__
    int width = EM_ASM_INT({
        return window.innerWidth;
    });

    int height = EM_ASM_INT({
        return window.innerHeight;
    });

    return {width, height};
#else

    initGLfw(/*headless*/ false);

    int count;
    auto monitors = glfwGetMonitors(&count);
    const GLFWvidmode* mode = glfwGetVideoMode(monitors[monitor]);

    return {mode->width, mode->height};
#endif
}

std::pair<float, float> monitor::contentScale(int monitor) {
#ifdef __EMSCRIPTEN__
    return {1, 1};//TODO
#else
    initGLfw(/*headless*/ false);

    int count;
    auto monitors = glfwGetMonitors(&count);

    float xscale, yscale;
    glfwGetMonitorContentScale(monitors[monitor], &xscale, &yscale);

    return {xscale, yscale};
#endif
}
