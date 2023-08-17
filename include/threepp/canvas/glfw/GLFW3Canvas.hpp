
#ifndef THREEPP_GLFW3CANVAS_HPP
#define THREEPP_GLFW3CANVAS_HPP

#include "threepp/canvas/Canvas.hpp"
#include "threepp/loaders/ImageLoader.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include <iostream>

namespace threepp {

    namespace detail {

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

    }// namespace detail

    class GLFW3Canvas: public CanvasBase {

    public:
        explicit GLFW3Canvas(const std::string& name)
            : GLFW3Canvas(CanvasOptions().title(name)) {}

        GLFW3Canvas(const std::string& name, const std::unordered_map<std::string, detail::ParameterValue>& values)
            : GLFW3Canvas(CanvasOptions(values).title(name)) {}

        explicit GLFW3Canvas(const CanvasOptions& params = CanvasOptions())
            : CanvasBase(params.size_) {

            glfwSetErrorCallback(error_callback);

            if (!glfwInit()) {
                exit(EXIT_FAILURE);
            }

            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
            glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
            glfwWindowHint(GLFW_RESIZABLE, params.resizable_);

            if (params.antialiasing_ > 0) {
                glfwWindowHint(GLFW_SAMPLES, params.antialiasing_);
            }

            window_ = glfwCreateWindow(params.size_.width, params.size_.height, params.title_.c_str(), nullptr, nullptr);
            if (!window_) {
                glfwTerminate();
                exit(EXIT_FAILURE);
            }

            setWindowIcon(window_, params.favicon_);

            glfwSetWindowUserPointer(window_, this);

            glfwSetKeyCallback(window_, key_callback);
            glfwSetMouseButtonCallback(window_, mouse_callback);
            glfwSetCursorPosCallback(window_, cursor_callback);
            glfwSetScrollCallback(window_, scroll_callback);
            glfwSetWindowSizeCallback(window_, window_size_callback);

            glfwMakeContextCurrent(window_);
            gladLoadGL();
            glfwSwapInterval(params.vsync_ ? 1 : 0);

            if (params.antialiasing_ > 0) {
                glEnable(GL_MULTISAMPLE);
            }

            glEnable(GL_PROGRAM_POINT_SIZE);
        }

        void setSize(WindowSize size) override {

            glfwSetWindowSize(window_, size.width, size.height);
        }

        bool animateOnce(const std::function<void()>& f) override {

            if (glfwWindowShouldClose(window_)) {
                return false;
            }

            handleTasks();

            f();

            glfwSwapBuffers(window_);
            glfwPollEvents();

            return true;
        }

        [[nodiscard]] GLFWwindow* windowPtr() const {

            return window_;
        }

        ~GLFW3Canvas() override {
            glfwDestroyWindow(window_);
            glfwTerminate();
        }

    protected:
        float getTime() override {

            return static_cast<float>(glfwGetTime());
        }

    private:
        GLFWwindow* window_;
        Vector2 lastMousePos_;

        static void window_size_callback(GLFWwindow* w, int width, int height) {
            auto p = static_cast<GLFW3Canvas*>(glfwGetWindowUserPointer(w));
            p->size_ = {width, height};
            if (p->resizeListener_) p->resizeListener_.value().operator()(p->size_);
        }

        static void error_callback(int error, const char* description) {
            std::cerr << "Error: " << description << std::endl;
        }

        static void scroll_callback(GLFWwindow* w, double xoffset, double yoffset) {
            auto p = static_cast<GLFW3Canvas*>(glfwGetWindowUserPointer(w));

            p->onMouseWheelEvent({static_cast<float>(xoffset), static_cast<float>(yoffset)});
        }

        static void mouse_callback(GLFWwindow* w, int button, int action, int) {
            auto p = static_cast<GLFW3Canvas*>(glfwGetWindowUserPointer(w));

            switch (action) {
                case GLFW_PRESS:
                    p->onMousePressedEvent(button, p->lastMousePos_, PeripheralsEventSource::MouseAction::PRESS);
                    break;
                case GLFW_RELEASE:
                    p->onMousePressedEvent(button, p->lastMousePos_, PeripheralsEventSource::MouseAction::RELEASE);
                    break;
                default:
                    break;
            }
        }

        static void cursor_callback(GLFWwindow* w, double xpos, double ypos) {
            auto p = static_cast<GLFW3Canvas*>(glfwGetWindowUserPointer(w));

            Vector2 mousePos(static_cast<float>(xpos), static_cast<float>(ypos));
            p->onMouseMoveEvent(mousePos);
            p->lastMousePos_.copy(mousePos);
        }

        static void key_callback(GLFWwindow* w, int key, int scancode, int action, int mods) {

            if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
                glfwSetWindowShouldClose(w, GLFW_TRUE);
                return;
            }

            KeyEvent evt{detail::glfwKeyCodeToKey(key), scancode, mods};
            auto p = static_cast<GLFW3Canvas*>(glfwGetWindowUserPointer(w));
            switch (action) {
                case GLFW_PRESS: {
                    p->onKeyEvent(evt, PeripheralsEventSource::KeyAction::PRESS);
                    break;
                }
                case GLFW_RELEASE: {
                    p->onKeyEvent(evt, PeripheralsEventSource::KeyAction::RELEASE);
                    break;
                }
                case GLFW_REPEAT: {
                    p->onKeyEvent(evt, PeripheralsEventSource::KeyAction::REPEAT);
                    break;
                }
                default:
                    break;
            }
        }

        static void setWindowIcon(GLFWwindow* window, std::optional<std::filesystem::path> customIcon) {

            ImageLoader imageLoader;
            std::optional<Image> favicon;
            if (customIcon) {
                favicon = imageLoader.load(*customIcon, 4, false);
            } else {
                favicon = loadFavicon();
            }
            if (favicon) {
                GLFWimage images[1];
                images[0] = {static_cast<int>(favicon->width),
                             static_cast<int>(favicon->height),
                             favicon->getData()};
                glfwSetWindowIcon(window, 1, images);
            }
        }
    };

}// namespace threepp

#endif//THREEPP_GLFW3CANVAS_HPP
