
#ifndef THREEPP_SDLCANVAS_HPP
#define THREEPP_SDLCANVAS_HPP

#include "threepp/canvas/Canvas.hpp"
#include "threepp/canvas/CanvasOptions.hpp"

#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <glad/glad.h>

#include <iostream>

#ifdef HAS_IMGUI
#include "imgui_impl_sdl2.h"
#endif

namespace threepp {

    namespace detail {

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

        inline Key getKey(SDL_Keycode key) {

            switch (key) {
                case SDLK_0:
                    return Key::NUM_0;
                case SDLK_1:
                    return Key::NUM_1;
                case SDLK_2:
                    return Key::NUM_2;
                case SDLK_3:
                    return Key::NUM_3;
                case SDLK_4:
                    return Key::NUM_4;
                case SDLK_5:
                    return Key::NUM_5;
                case SDLK_6:
                    return Key::NUM_6;
                case SDLK_7:
                    return Key::NUM_7;
                case SDLK_8:
                    return Key::NUM_8;
                case SDLK_9:
                    return Key::NUM_9;

                case SDLK_a:
                    return Key::A;
                case SDLK_b:
                    return Key::B;
                case SDLK_c:
                    return Key::C;
                case SDLK_d:
                    return Key::D;
                case SDLK_e:
                    return Key::E;
                case SDLK_f:
                    return Key::F;
                case SDLK_g:
                    return Key::G;
                case SDLK_h:
                    return Key::H;
                case SDLK_j:
                    return Key::J;
                case SDLK_k:
                    return Key::K;
                case SDLK_l:
                    return Key::L;
                case SDLK_m:
                    return Key::M;
                case SDLK_n:
                    return Key::N;
                case SDLK_o:
                    return Key::O;
                case SDLK_p:
                    return Key::P;
                case SDLK_r:
                    return Key::R;
                case SDLK_s:
                    return Key::S;
                case SDLK_t:
                    return Key::T;
                case SDLK_u:
                    return Key::U;
                case SDLK_v:
                    return Key::V;
                case SDLK_w:
                    return Key::W;
                case SDLK_x:
                    return Key::X;
                case SDLK_y:
                    return Key::Y;
                case SDLK_z:
                    return Key::Z;

                case SDLK_UP:
                    return Key::UP;
                case SDLK_DOWN:
                    return Key::DOWN;
                case SDLK_LEFT:
                    return Key::LEFT;
                case SDLK_RIGHT:
                    return Key::RIGHT;

                case SDLK_SPACE:
                    return Key::SPACE;

                default:
                    return Key::UNKNOWN;
            }
        }
    }// namespace detail

    class SDLCanvas: public CanvasBase {

    public:
        explicit SDLCanvas(const std::string& name)
            : SDLCanvas(CanvasOptions().title(name)) {}

        SDLCanvas(const std::string& name, const std::unordered_map<std::string, detail::ParameterValue>& values)
            : SDLCanvas(CanvasOptions(values).title(name)) {}

        explicit SDLCanvas(const CanvasOptions& params = CanvasOptions())
            : CanvasBase(params.size_), event{} {

            if (SDL_Init(SDL_INIT_VIDEO) < 0) {
                detail::sdldie("");
            }
            SDL_GL_LoadLibrary(nullptr);// Default OpenGL is fine.

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
                detail::sdldie("");
            }
            maincontext = SDL_GL_CreateContext(window);

            //            setWindowIcon(window, params.favicon_);

            gladLoadGL();

            SDL_GL_SetSwapInterval(params.vsync_ ? 1 : 0);

            if (params.antialiasing_ > 0) {
                glEnable(GL_MULTISAMPLE);
            }

            glEnable(GL_PROGRAM_POINT_SIZE);
        }

        void setSize(WindowSize size) override {
            SDL_SetWindowSize(window, size.width, size.height);
        }

        bool animateOnce(const std::function<void()>& f) override {

            SDL_GL_SwapWindow(window);

            if (processEvents) {
                while (SDL_PollEvent(&event)) {
                    if (event.type == SDL_QUIT) {
                        return false;
                    }
                    handleEvents();
                }
            }

            handleTasks();

            f();

            return true;
        }

    protected:
        float getTime() override {
            return static_cast<float>(SDL_GetTicks()) / 1000;
        }

#ifdef HAS_IMGUI
        void initImguiContext() override {
            ImGui_ImplSDL2_InitForOpenGL(window, &maincontext);
        }

        void newImguiFrame() override {
            processEvents = !ImGui_ImplSDL2_ProcessEvent(&event);
            ImGui_ImplSDL2_NewFrame();
        }

        void destroyImguiContext() override {
            ImGui_ImplSDL2_Shutdown();
        }
#endif

    private:
        SDL_Event event;
        SDL_Window* window;
        SDL_GLContext maincontext;

        bool processEvents{true};

        void handleEvents() {

            if (event.type == SDL_WINDOWEVENT) {
                if (event.window.event == SDL_WINDOWEVENT_RESIZED && resizeListener_) {
                    WindowSize s{};
                    SDL_GetWindowSize(window, &s.width, &s.height);
                    resizeListener_->operator()(s);
                }
            } else if (event.type == SDL_MOUSEBUTTONDOWN) {

                auto action = event.button;
                onMousePressedEvent(detail::sdlButtonConvert(action.button), {action.x, action.y}, MouseAction::PRESS);

            } else if (event.type == SDL_MOUSEBUTTONUP) {

                auto action = event.button;
                onMousePressedEvent(detail::sdlButtonConvert(action.button), {action.x, action.y}, MouseAction::RELEASE);


            } else if (event.type == SDL_MOUSEMOTION) {

                auto action = event.button;
                onMouseMoveEvent({action.x, action.y});

            } else if (event.type == SDL_MOUSEWHEEL) {

                auto action = event.wheel;
                onMouseWheelEvent({action.x, action.y});


            } else if (event.type == SDL_KEYDOWN) {

                KeyEvent evt{detail::getKey(event.key.keysym.sym), event.key.keysym.scancode, event.key.keysym.mod};

                if (event.key.repeat) {
                    onKeyEvent(evt, KeyAction::REPEAT);
                } else {
                    onKeyEvent(evt, KeyAction::PRESS);
                }

            } else if (event.type == SDL_KEYUP) {

                KeyEvent evt{detail::getKey(event.key.keysym.sym), event.key.keysym.scancode, event.key.keysym.mod};
                onKeyEvent(evt, KeyAction::RELEASE);
            }
        }
    };

}// namespace threepp

#endif//THREEPP_SDLCANVAS_HPP
