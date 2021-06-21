// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLState.js

#ifndef THREEPP_GLSTATE_HPP
#define THREEPP_GLSTATE_HPP

#include <utility>

#include "glad/glad.h"

#include "threepp/math/Vector4.hpp"

namespace threepp::gl {

    struct GLState {


        struct GLColorBuffer {

            bool locked = false;

            Vector4 color;
            std::optional<bool> currentColorMask;
            Vector4 currentColorClear = Vector4(0, 0, 0, 0);

            void setMask(bool colorMask) {

                if (!locked && currentColorMask && currentColorMask.value() != colorMask) {

                    glColorMask(colorMask, colorMask, colorMask, colorMask);
                    currentColorMask = colorMask;
                }
            }

            void setLocked(bool lock) {

                locked = lock;
            }

            void setClear(float r, float g, float b, float a, bool premultipliedAlpha) {

                if (premultipliedAlpha) {

                    r *= a;
                    g *= a;
                    b *= a;
                }

                color.set(r, g, b, a);

                if (!currentColorClear.equals(color)) {

                    glClearColor(r, g, b, a);
                    currentColorClear.copy(color);
                }
            }

            void reset() {

                locked = false;

                currentColorMask = std::nullopt;
                currentColorClear.set(-1, 0, 0, 0);// set to invalid state
            }
        };

        struct DepthBuffer {

            bool locked = false;

            std::optional<bool> currentDepthMask;
            std::optional<int> currentDepthFunc;
            std::optional<float> currentDepthClear;

            std::function<void(int)> enable;
            std::function<void(int)> disable;

            DepthBuffer(std::function<void(int)> enable, std::function<void(int)> disable) : enable(std::move(enable)), disable(std::move(disable)) {}

            void setTest(bool depthTest) {

                if (depthTest) {

                    enable(GL_DEPTH_TEST);

                } else {

                    disable(GL_DEPTH_TEST);
                }
            }

            void setMask(bool depthMask) {

                if (!locked && currentDepthMask && currentDepthMask.value() != depthMask) {

                    glDepthMask(depthMask);
                    currentDepthMask = depthMask;
                }
            }

            void setFunc(int depthFunc) {

                if (currentDepthFunc && currentDepthFunc.value() != depthFunc) {

                    switch (depthFunc) {

                        case NeverDepth:

                            glDepthFunc(GL_NEVER);
                            break;

                        case AlwaysDepth:

                            glDepthFunc(GL_ALWAYS);
                            break;

                        case LessDepth:

                            glDepthFunc(GL_LESS);
                            break;

                        case LessEqualDepth:

                            glDepthFunc(GL_LEQUAL);
                            break;

                        case EqualDepth:

                            glDepthFunc(GL_EQUAL);
                            break;

                        case GreaterEqualDepth:

                            glDepthFunc(GL_GEQUAL);
                            break;

                        case GreaterDepth:

                            glDepthFunc(GL_GREATER);
                            break;

                        case NotEqualDepth:

                            glDepthFunc(GL_NOTEQUAL);
                            break;

                        default:

                            glDepthFunc(GL_LEQUAL);
                    }

                    currentDepthFunc = depthFunc;
                }
            }

            void setLocked(bool lock) {

                locked = lock;
            }

            void setClear(float depth) {
                if (currentDepthClear != depth) {

                    glClearDepth(depth);
                    currentDepthClear = depth;
                }
            }

            void reset() {

                locked = false;

                currentDepthMask = std::nullopt;
                currentDepthFunc = std::nullopt;
                currentDepthClear = std::nullopt;
            }
        };

        GLState() {}
    };

}// namespace threepp::gl

#endif//THREEPP_GLSTATE_HPP
