// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLState.js

#ifndef THREEPP_GLSTATE_HPP
#define THREEPP_GLSTATE_HPP

#include <iostream>
#include <utility>

#include "glad/glad.h"

#include "threepp/materials/Material.hpp"
#include "threepp/math/Vector4.hpp"


namespace threepp::gl {

    GLenum equationToGL(int eq) {

        switch (eq) {
            case AddEquation:
                return GL_FUNC_ADD;
            case SubtractEquation:
                return GL_FUNC_SUBTRACT;
            case ReverseSubtractEquation:
                return GL_FUNC_REVERSE_SUBTRACT;
            case MinEquation:
                return GL_MIN;
            case MaxEquation:
                return GL_MAX;
            default:
                throw std::runtime_error("Unknown equation: " + std::to_string(eq));
        }
    }

    GLenum factorToGL(int factor) {

        switch (factor) {
            case ZeroFactor:
                return GL_ZERO;
            case OneFactor:
                return GL_ONE;
            case SrcColorFactor:
                return GL_SRC_COLOR;
            case SrcAlphaFactor:
                return GL_SRC_ALPHA;
            case SrcAlphaSaturateFactor:
                return GL_SRC_ALPHA_SATURATE;
            case DstColorFactor:
                return GL_DST_COLOR;
            case DstAlphaFactor:
                return GL_DST_ALPHA;
            case OneMinusSrcColorFactor:
                return GL_ONE_MINUS_SRC_COLOR;
            case OneMinusSrcAlphaFactor:
                return GL_ONE_MINUS_SRC_ALPHA;
            case OneMinusDstColorFactor:
                return GL_ONE_MINUS_DST_COLOR;
            case OneMinusDstAlphaFactor:
                return GL_ONE_MINUS_DST_ALPHA;
            default:
                throw std::runtime_error("Unknown factor: " + std::to_string(factor));
        }
    }

    struct BoundTexture {

        std::optional<int> type;
        std::optional<int> texture;
    };


    struct ColorBuffer {

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

        void setClear(float r, float g, float b, float a, bool premultipliedAlpha = false) {

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

        void setFunctions(std::function<void(int)> enable, std::function<void(int)> disable) {
            this->enable = std::move(enable);
            this->disable = std::move(disable);
        }

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

    struct StencilBuffer {

        bool locked = false;

        std::optional<int> currentStencilMask;
        std::optional<int> currentStencilFunc;
        std::optional<int> currentStencilRef;
        std::optional<int> currentStencilFuncMask;
        std::optional<int> currentStencilFail;
        std::optional<int> currentStencilZFail;
        std::optional<int> currentStencilZPass;
        std::optional<int> currentStencilClear;

        std::function<void(int)> enable;
        std::function<void(int)> disable;

        void setFunctions(std::function<void(int)> enable, std::function<void(int)> disable) {
            this->enable = std::move(enable);
            this->disable = std::move(disable);
        }

        void setTest(bool stencilTest) {

            if (!locked) {

                if (stencilTest) {

                    enable(GL_STENCIL_TEST);

                } else {

                    disable(GL_STENCIL_TEST);
                }
            }
        }

        void setMask(int stencilMask) {

            if (!locked && currentStencilMask && currentStencilMask.value() != stencilMask) {

                glStencilMask(stencilMask);
                currentStencilMask = stencilMask;
            }
        }

        void setFunc(int stencilFunc, int stencilRef, int stencilMask) {

            if (currentStencilFunc != stencilFunc ||
                currentStencilRef != stencilRef ||
                currentStencilFuncMask != stencilMask) {

                glStencilFunc(stencilFunc, stencilRef, stencilMask);

                currentStencilFunc = stencilFunc;
                currentStencilRef = stencilRef;
                currentStencilFuncMask = stencilMask;
            }
        }

        void setOp(int stencilFail, int stencilZFail, int stencilZPass) {

            if (currentStencilFail != stencilFail ||
                currentStencilZFail != stencilZFail ||
                currentStencilZPass != stencilZPass) {

                glStencilOp(stencilFail, stencilZFail, stencilZPass);

                currentStencilFail = stencilFail;
                currentStencilZFail = stencilZFail;
                currentStencilZPass = stencilZPass;
            }
        }

        void setLocked(bool lock) {

            locked = lock;
        }

        void setClear(int stencil) {

            if (currentStencilClear != stencil) {

                glClearStencil(stencil);
                currentStencilClear = stencil;
            }
        }

        void reset() {

            locked = false;

            currentStencilMask = std::nullopt;
            currentStencilFunc = std::nullopt;
            currentStencilRef = std::nullopt;
            currentStencilFuncMask = std::nullopt;
            currentStencilFail = std::nullopt;
            currentStencilZFail = std::nullopt;
            currentStencilZPass = std::nullopt;
            currentStencilClear = std::nullopt;
        }
    };


    struct GLState {

        ColorBuffer colorBuffer;
        DepthBuffer depthBuffer;
        StencilBuffer stencilBuffer;

        std::unordered_map<int, bool> enabledCapabilities;

        std::optional<int> xrFramebuffer;
        std::unordered_map<int, int> currentBoundFramebuffers;

        std::optional<int> currentProgram;

        bool currentBlendingEnabled = false;
        std::optional<int> currentBlending;
        std::optional<int> currentBlendEquation;
        std::optional<int> currentBlendSrc;
        std::optional<int> currentBlendDst;
        std::optional<int> currentBlendEquationAlpha;
        std::optional<int> currentBlendSrcAlpha;
        std::optional<int> currentBlendDstAlpha;
        std::optional<bool> currentPremultipledAlpha = false;

        std::optional<bool> currentFlipSided;
        std::optional<int> currentCullFace;

        std::optional<float> currentLineWidth;

        std::optional<float> currentPolygonOffsetFactor;
        std::optional<float> currentPolygonOffsetUnits;

        GLint64 maxTextures;

        bool lineWidthAvailable = false;
        unsigned int version = 0;

        std::optional<int> currentTextureSlot;
        std::unordered_map<int, BoundTexture> currentBoundTextures;

        std::unordered_map<int, int> emptyTextures;

        Vector4 currentScissor;
        Vector4 currentViewport;

        const Canvas &canvas;

        GLState(const Canvas &canvas) : canvas(canvas) {

            glGetInteger64v(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &maxTextures);

            GLint64 *scissorParam;
            GLint64 *viewportParam;
            glGetInteger64v(GL_SCISSOR_BOX, scissorParam);
            glGetInteger64v(GL_VIEWPORT, viewportParam);

            currentScissor.fromArray(scissorParam);
            currentViewport.fromArray(scissorParam);

            delete scissorParam;
            delete viewportParam;

            auto enableLambda = [&](int id) {
                enable(id);
            };

            auto disableLambda = [&](int id) {
                disable(id);
            };

            std::function<GLuint(GLenum, GLenum, int)> createTexture = [](GLenum type, GLenum target, int count) {
                GLint64 data[4];// 4 is required to match default unpack alignment of 4.
                GLuint textureArray[1];
                glGenTextures(1, textureArray);

                GLuint texture = textureArray[0];

                glBindTexture(type, texture);
                glTexParameteri(type, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                glTexParameteri(type, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

                for (int i = 0; i < count; i++) {
                    glTexImage2D(target + 1, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
                }

                return texture;
            };

            depthBuffer.setFunctions(enableLambda, disableLambda);
            stencilBuffer.setFunctions(enableLambda, disableLambda);

            emptyTextures[GL_TEXTURE_2D] = createTexture(GL_TEXTURE_2D, GL_TEXTURE_2D, 1);
            emptyTextures[GL_TEXTURE_CUBE_MAP] = createTexture(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_CUBE_MAP, 1);

            colorBuffer.setClear(0, 0, 0, 1);
            depthBuffer.setClear(1);
            stencilBuffer.setClear(1);

            enable(GL_DEPTH_TEST);
            depthBuffer.setFunc(LessEqualDepth);

            enable(GL_CULL_FACE);
        }

        void enable(int id) {
            if (!enabledCapabilities[id]) {

                glEnable(id);
                enabledCapabilities[id] = true;
            }
        }

        void disable(int id) {
            if (enabledCapabilities[id]) {

                glDisable(id);
                enabledCapabilities[id] = false;
            }
        }

        void bindXRFramebuffer(int framebuffer) {

            if (framebuffer != xrFramebuffer) {

                glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

                xrFramebuffer = framebuffer;
            }
        }

        bool bindFramebuffer(int target, int framebuffer) {

            if (!framebuffer && xrFramebuffer) framebuffer = *xrFramebuffer;// use active XR framebuffer if available

            if (currentBoundFramebuffers.count(target) && currentBoundFramebuffers[target] != framebuffer) {

                glBindFramebuffer(target, framebuffer);

                currentBoundFramebuffers[target] = framebuffer;

                // GL_DRAW_FRAMEBUFFER is equivalent to GL_FRAMEBUFFER

                if (target == GL_DRAW_FRAMEBUFFER) {

                    currentBoundFramebuffers[GL_FRAMEBUFFER] = framebuffer;
                }

                if (target == GL_FRAMEBUFFER) {

                    currentBoundFramebuffers[GL_DRAW_FRAMEBUFFER] = framebuffer;
                }


                return true;
            }

            return false;
        }

        bool useProgram(int program) {

            if (currentProgram != program) {

                glUseProgram(program);

                currentProgram = program;

                return true;
            }

            return false;
        }

        void setBlending(
                int blending,
                std::optional<int> blendEquation = std::nullopt,
                std::optional<int> blendSrc = std::nullopt,
                std::optional<int> blendDst = std::nullopt,
                std::optional<int> blendEquationAlpha = std::nullopt,
                std::optional<int> blendSrcAlpha = std::nullopt,
                std::optional<int> blendDstAlpha = std::nullopt,
                std::optional<bool> premultipliedAlpha = std::nullopt) {

            if (blending == NoBlending) {

                if (currentBlendingEnabled) {

                    disable(GL_BLEND);
                    currentBlendingEnabled = false;
                }

                return;
            }

            if (currentBlendingEnabled) {

                enable(GL_BLEND);
                currentBlendingEnabled = true;
            }

            if (blending != CustomBlending) {

                if (blending != currentBlending || premultipliedAlpha != currentPremultipledAlpha) {

                    if (currentBlendEquation != AddEquation || currentBlendEquationAlpha != AddEquation) {

                        glBlendEquation(GL_FUNC_ADD);

                        currentBlendEquation = AddEquation;
                        currentBlendEquationAlpha = AddEquation;
                    }

                    if (premultipliedAlpha) {

                        switch (blending) {

                            case NormalBlending:
                                glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
                                break;

                            case AdditiveBlending:
                                glBlendFunc(GL_ONE, GL_ONE);
                                break;

                            case SubtractiveBlending:
                                glBlendFuncSeparate(GL_ZERO, GL_ZERO, GL_ONE_MINUS_SRC_COLOR, GL_ONE_MINUS_SRC_ALPHA);
                                break;

                            case MultiplyBlending:
                                glBlendFuncSeparate(GL_ZERO, GL_SRC_COLOR, GL_ZERO, GL_SRC_ALPHA);
                                break;

                            default:
                                std::cerr << "THREE.WebGLState: Invalid blending: " << blending << std::endl;
                                break;
                        }

                    } else {

                        switch (blending) {

                            case NormalBlending:
                                glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
                                break;

                            case AdditiveBlending:
                                glBlendFunc(GL_SRC_ALPHA, GL_ONE);
                                break;

                            case SubtractiveBlending:
                                glBlendFunc(GL_ZERO, GL_ONE_MINUS_SRC_COLOR);
                                break;

                            case MultiplyBlending:
                                glBlendFunc(GL_ZERO, GL_SRC_COLOR);
                                break;

                            default:
                                std::cerr << "THREE.WebGLState: Invalid blending: " << blending << std::endl;
                                break;
                        }
                    }

                    currentBlendSrc = std::nullopt;
                    currentBlendDst = std::nullopt;
                    currentBlendSrcAlpha = std::nullopt;
                    currentBlendDstAlpha = std::nullopt;

                    currentBlending = blending;
                    currentPremultipledAlpha = premultipliedAlpha;
                }

                return;
            }

            // custom blending

            blendEquationAlpha = blendEquationAlpha || blendEquation;
            blendSrcAlpha = blendSrcAlpha || blendSrc;
            blendDstAlpha = blendDstAlpha || blendDst;

            if (blendEquation != currentBlendEquation || blendEquationAlpha != currentBlendEquationAlpha) {

                glBlendEquationSeparate(equationToGL(*blendEquation), equationToGL(*blendEquationAlpha));

                currentBlendEquation = blendEquation;
                currentBlendEquationAlpha = blendEquationAlpha;
            }

            if (blendSrc != currentBlendSrc || blendDst != currentBlendDst || blendSrcAlpha != currentBlendSrcAlpha || blendDstAlpha != currentBlendDstAlpha) {

                glBlendFuncSeparate(factorToGL(*blendSrc), factorToGL(*blendDst), factorToGL(*blendSrcAlpha), factorToGL(*blendDstAlpha));

                currentBlendSrc = blendSrc;
                currentBlendDst = blendDst;
                currentBlendSrcAlpha = blendSrcAlpha;
                currentBlendDstAlpha = blendDstAlpha;
            }

            currentBlending = blending;
            currentPremultipledAlpha = std::nullopt;
        }


        void setMaterial(const Material &material, bool frontFaceCW) {

            material.side == DoubleSide
                    ? disable(GL_CULL_FACE)
                    : enable(GL_CULL_FACE);

            auto flipSided = (material.side == BackSide);
            if (frontFaceCW) flipSided = !flipSided;

            setFlipSided(flipSided);

            (material.blending == NormalBlending && !material.transparent)
                    ? setBlending(NoBlending)
                    : setBlending(material.blending, material.blendEquation, material.blendSrc, material.blendDst, material.blendEquationAlpha, material.blendSrcAlpha, material.blendDstAlpha, material.premultipliedAlpha);

            depthBuffer.setFunc(material.depthFunc);
            depthBuffer.setTest(material.depthTest);
            depthBuffer.setMask(material.depthWrite);
            colorBuffer.setMask(material.colorWrite);

            const auto stencilWrite = material.stencilWrite;
            stencilBuffer.setTest(stencilWrite);
            if (stencilWrite) {

                stencilBuffer.setMask(material.stencilWriteMask);
                stencilBuffer.setFunc(material.stencilFunc, material.stencilRef, material.stencilFuncMask);
                stencilBuffer.setOp(material.stencilFail, material.stencilZFail, material.stencilZPass);
            }

            setPolygonOffset(material.polygonOffset, material.polygonOffsetFactor, material.polygonOffsetUnits);

            material.alphaToCoverage ? enable(GL_SAMPLE_ALPHA_TO_COVERAGE) : disable(GL_SAMPLE_ALPHA_TO_COVERAGE);
        }

        void setFlipSided(bool flipSided) {

            if (currentFlipSided != flipSided) {

                if (flipSided) {

                    glFrontFace(GL_CW);

                } else {

                    glFrontFace(GL_CCW);
                }

                currentFlipSided = flipSided;
            }
        }

        void setCullFace(int cullFace) {

            if (cullFace != CullFaceNone) {

                enable(GL_CULL_FACE);

                if (cullFace != currentCullFace) {

                    if (cullFace == CullFaceBack) {

                        glCullFace(GL_BACK);

                    } else if (cullFace == CullFaceFront) {

                        glCullFace(GL_FRONT);

                    } else {

                        glCullFace(GL_FRONT_AND_BACK);
                    }
                }

            } else {

                disable(GL_CULL_FACE);
            }

            currentCullFace = cullFace;
        }

        void setLineWidth(float width) {

            if (width != currentLineWidth) {

                if (lineWidthAvailable) glLineWidth(width);

                currentLineWidth = width;
            }
        }

        void setPolygonOffset(bool polygonOffset, float factor, float units) {

            if (polygonOffset) {

                enable(GL_POLYGON_OFFSET_FILL);

                if (currentPolygonOffsetFactor != factor || currentPolygonOffsetUnits != units) {

                    glPolygonOffset(factor, units);

                    currentPolygonOffsetFactor = factor;
                    currentPolygonOffsetUnits = units;
                }

            } else {

                disable(GL_POLYGON_OFFSET_FILL);
            }
        }

        void setScissorTest(bool scissorTest) {

            if (scissorTest) {

                enable(GL_SCISSOR_TEST);

            } else {

                disable(GL_SCISSOR_TEST);
            }
        }

        // texture

        void activeTexture(std::optional<GLenum> glSlot = std::nullopt) {

            if (!glSlot) glSlot = (GLenum) (GL_TEXTURE0 + maxTextures - 1);

            if (currentTextureSlot != glSlot) {

                glActiveTexture(*glSlot);
                currentTextureSlot = *glSlot;
            }
        }

        void bindTexture(int glType, int glTexture) {

            if (!currentTextureSlot) {

                activeTexture();
            }

            BoundTexture &boundTexture = currentBoundTextures[*currentTextureSlot];

            //            if ( boundTexture === undefined ) {
            //
            //                boundTexture = { type: undefined, texture: undefined };
            //                currentBoundTextures[ currentTextureSlot ] = boundTexture;
            //
            //            }

            if (boundTexture.type != glType || boundTexture.texture != glTexture) {

                glBindTexture(glType, glTexture || emptyTextures[glType]);

                boundTexture.type = glType;
                boundTexture.texture = glTexture;
            }
        }

        void unbindTexture() {

            if (currentTextureSlot && currentBoundTextures.count(*currentTextureSlot)) {

                BoundTexture &boundTexture = currentBoundTextures.at(*currentTextureSlot);

                if (boundTexture.type) {

                    glBindTexture(*boundTexture.type, 0);

                    boundTexture.type = std::nullopt;
                    boundTexture.texture = std::nullopt;
                }
            }
        }

        void texImage2D(
                int target,
                int level,
                int internalFormat,
                int width,
                int height,
                int format,
                int type,
                const void *pixels) {

            glTexImage2D(target, level, internalFormat, width, height, 0, format, type, pixels);
        }

        //

        void scissor(const Vector4 &scissor) {

            if (!currentScissor.equals(scissor)) {

                glScissor((GLint) scissor.x, (GLint) scissor.y, (GLsizei) scissor.z, (GLsizei) scissor.w);
                currentScissor.copy(scissor);
            }
        }

        void viewport(const Vector4 &viewport) {

            if (!currentViewport.equals(viewport)) {

                glViewport((GLint) viewport.x, (GLint) viewport.y, (GLsizei) viewport.z, (GLsizei) viewport.w);
                currentViewport.copy(viewport);
            }
        }

        //

        void reset() {

            // reset state

            glDisable(GL_BLEND);
            glDisable(GL_CULL_FACE);
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_POLYGON_OFFSET_FILL);
            glDisable(GL_SCISSOR_TEST);
            glDisable(GL_STENCIL_TEST);
            glDisable(GL_SAMPLE_ALPHA_TO_COVERAGE);

            glBlendEquation(GL_FUNC_ADD);
            glBlendFunc(GL_ONE, GL_ZERO);
            glBlendFuncSeparate(GL_ONE, GL_ZERO, GL_ONE, GL_ZERO);

            glColorMask(true, true, true, true);
            glClearColor(0, 0, 0, 0);

            glDepthMask(true);
            glDepthFunc(GL_LESS);
            glClearDepth(1);

            glStencilMask(0xffffffff);
            glStencilFunc(GL_ALWAYS, 0, 0xffffffff);
            glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
            glClearStencil(0);

            glCullFace(GL_BACK);
            glFrontFace(GL_CCW);

            glPolygonOffset(0, 0);

            glActiveTexture(GL_TEXTURE0);

            glBindFramebuffer(GL_FRAMEBUFFER, 0);

            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

            glUseProgram(0);

            glLineWidth(1);

            glScissor(0, 0, canvas.getWidth(), canvas.getHeight());
            glViewport(0, 0, canvas.getWidth(), canvas.getHeight());

            // reset internals

            enabledCapabilities.clear();

            currentTextureSlot = std::nullopt;
            currentBoundTextures.clear();

            xrFramebuffer = std::nullopt;
            currentBoundFramebuffers.clear();

            currentProgram = std::nullopt;

            currentBlendingEnabled = false;
            currentBlending = std::nullopt;
            currentBlendEquation = std::nullopt;
            currentBlendSrc = std::nullopt;
            currentBlendDst = std::nullopt;
            currentBlendEquationAlpha = std::nullopt;
            currentBlendSrcAlpha = std::nullopt;
            currentBlendDstAlpha = std::nullopt;
            currentPremultipledAlpha = false;

            currentFlipSided = std::nullopt;
            currentCullFace = std::nullopt;

            currentLineWidth = std::nullopt;

            currentPolygonOffsetFactor = std::nullopt;
            currentPolygonOffsetUnits = std::nullopt;

            currentScissor.set(0, 0, (float) canvas.getWidth(), (float) canvas.getHeight());
            currentViewport.set(0, 0, (float) canvas.getWidth(), (float) canvas.getHeight());

            colorBuffer.reset();
            depthBuffer.reset();
            stencilBuffer.reset();
        }
    };

}// namespace threepp::gl

#endif//THREEPP_GLSTATE_HPP
