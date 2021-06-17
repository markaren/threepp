
#ifndef THREEPP_CANVAS_HPP
#define THREEPP_CANVAS_HPP

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include <functional>
#include <utility>

#include "threepp/core/Clock.hpp"

namespace threepp {

    class Canvas {

    public:

        struct Parameters {

            int width_ = 640;
            int height_ = 480;

            std::string title_;

            Parameters() = default;

            Parameters &title(std::string value) {

                this->title_ = std::move(value);

                return *this;
            }

            Parameters &width(unsigned int value) {

                this->width_ = value;

                return *this;
            }

            Parameters &height(unsigned int value) {

                this->height_ = value;

                return *this;
            }

        };

        explicit Canvas(const Parameters& params) {
            glfwSetErrorCallback(error_callback);

            if (!glfwInit()) {
                exit(EXIT_FAILURE);
            }

            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

            window = glfwCreateWindow(params.width_, params.height_, params.title_.c_str(), nullptr, nullptr);
            if (!window) {
                glfwTerminate();
                exit(EXIT_FAILURE);
            }

            glfwSetKeyCallback(window, key_callback);

            glfwMakeContextCurrent(window);
            gladLoadGL();
            glfwSwapInterval(1);
        }

        void animate(const std::function<void(float)>& f) {

            Clock clock;
            while (!glfwWindowShouldClose(window)) {

                f(clock.getDelta());

                glfwSwapBuffers(window);
                glfwPollEvents();
            }
        }

        ~Canvas() {

            glfwDestroyWindow(window);
            glfwTerminate();
        }


    private:
        GLFWwindow *window;

        static void error_callback(int error, const char *description) {
            fprintf(stderr, "Error: %s\n", description);
        }

        static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
            if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
                glfwSetWindowShouldClose(window, GLFW_TRUE);
        }


    };

}// namespace threepp

#endif//THREEPP_CANVAS_HPP
