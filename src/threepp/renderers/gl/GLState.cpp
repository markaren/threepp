
#include "threepp/renderers/gl/GLState.hpp"

#include "threepp/materials/Material.hpp"
#include "threepp/renderers/gl/GLUtils.hpp"

#if EMSCRIPTEN
#include <GLES3/gl32.h>
#endif

#include <iostream>

using namespace threepp;

namespace {

    GLenum equationToGL(BlendEquation eq) {

        switch (eq) {
            case BlendEquation::Add:
                return GL_FUNC_ADD;
            case BlendEquation::Subtract:
                return GL_FUNC_SUBTRACT;
            case BlendEquation::ReverseSubtract:
                return GL_FUNC_REVERSE_SUBTRACT;
            case BlendEquation::Min:
                return GL_MIN;
            case BlendEquation::Max:
                return GL_MAX;
            default:
                throw std::runtime_error("Unknown equation: " + std::to_string(as_integer(eq)));
        }
    }

    GLenum factorToGL(BlendFactor factor) {

        switch (factor) {
            case BlendFactor::Zero:
                return GL_ZERO;
            case BlendFactor::One:
                return GL_ONE;
            case BlendFactor::SrcColor:
                return GL_SRC_COLOR;
            case BlendFactor::SrcAlpha:
                return GL_SRC_ALPHA;
            case BlendFactor::SrcAlphaSaturate:
                return GL_SRC_ALPHA_SATURATE;
            case BlendFactor::DstColor:
                return GL_DST_COLOR;
            case BlendFactor::DstAlpha:
                return GL_DST_ALPHA;
            case BlendFactor::OneMinusSrcColor:
                return GL_ONE_MINUS_SRC_COLOR;
            case BlendFactor::OneMinusSrcAlpha:
                return GL_ONE_MINUS_SRC_ALPHA;
            case BlendFactor::OneMinusDstColor:
                return GL_ONE_MINUS_DST_COLOR;
            case BlendFactor::OneMinusDstAlpha:
                return GL_ONE_MINUS_DST_ALPHA;
            default:
                throw std::runtime_error("Unknown factor: " + std::to_string(as_integer(factor)));
        }
    }

}// namespace

void gl::ColorBuffer::setMask(bool colorMask) {

    if (!locked && currentColorMask != colorMask) {

        glColorMask(colorMask, colorMask, colorMask, colorMask);
        currentColorMask = colorMask;
    }
}

void gl::ColorBuffer::setLocked(bool lock) {

    locked = lock;
}

void gl::ColorBuffer::setClear(float r, float g, float b, float a, bool premultipliedAlpha) {

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

void gl::ColorBuffer::reset() {

    locked = false;

    currentColorMask = std::nullopt;
    currentColorClear.set(-1, 0, 0, 0);// set to invalid state
}

void gl::DepthBuffer::setFunctions(std::function<void(int)> enable, std::function<void(int)> disable) {
    this->enable = std::move(enable);
    this->disable = std::move(disable);
}

void gl::DepthBuffer::setTest(bool depthTest) {

    if (depthTest) {

        enable(GL_DEPTH_TEST);

    } else {

        disable(GL_DEPTH_TEST);
    }
}

void gl::DepthBuffer::setMask(bool depthMask) {

    if (!locked && currentDepthMask != depthMask) {

        glDepthMask(depthMask);
        currentDepthMask = depthMask;
    }
}

void gl::DepthBuffer::setFunc(DepthFunc depthFunc) {

    if (currentDepthFunc != depthFunc) {

        switch (depthFunc) {

            case DepthFunc::Never:

                glDepthFunc(GL_NEVER);
                break;

            case DepthFunc::Always:

                glDepthFunc(GL_ALWAYS);
                break;

            case DepthFunc::Less:

                glDepthFunc(GL_LESS);
                break;

            case DepthFunc::LessEqual:

                glDepthFunc(GL_LEQUAL);
                break;

            case DepthFunc::Equal:

                glDepthFunc(GL_EQUAL);
                break;

            case DepthFunc::GreaterEqual:

                glDepthFunc(GL_GEQUAL);
                break;

            case DepthFunc::Greater:

                glDepthFunc(GL_GREATER);
                break;

            case DepthFunc::NotEqual:

                glDepthFunc(GL_NOTEQUAL);
                break;

            default:

                glDepthFunc(GL_LEQUAL);
        }

        currentDepthFunc = depthFunc;
    }
}

void gl::DepthBuffer::setLocked(bool lock) {

    locked = lock;
}

void gl::DepthBuffer::setClear(float depth) {
    if (currentDepthClear != depth) {

        glClearDepth(depth);
        currentDepthClear = depth;
    }
}

void gl::DepthBuffer::reset() {

    locked = false;

    currentDepthMask = std::nullopt;
    currentDepthFunc = std::nullopt;
    currentDepthClear = std::nullopt;
}

void gl::StencilBuffer::setFunctions(std::function<void(int)> enable, std::function<void(int)> disable) {
    this->enable = std::move(enable);
    this->disable = std::move(disable);
}

void gl::StencilBuffer::setTest(bool stencilTest) {

    if (!locked) {

        if (stencilTest) {

            enable(GL_STENCIL_TEST);

        } else {

            disable(GL_STENCIL_TEST);
        }
    }
}

void gl::StencilBuffer::setMask(int stencilMask) {

    if (!locked && currentStencilMask != stencilMask) {

        glStencilMask(stencilMask);
        currentStencilMask = stencilMask;
    }
}

void gl::StencilBuffer::setFunc(StencilFunc stencilFunc, int stencilRef, int stencilMask) {

    if (currentStencilFunc != stencilFunc ||
        currentStencilRef != stencilRef ||
        currentStencilFuncMask != stencilMask) {

        glStencilFunc(as_integer(stencilFunc), stencilRef, stencilMask);

        currentStencilFunc = stencilFunc;
        currentStencilRef = stencilRef;
        currentStencilFuncMask = stencilMask;
    }
}

void gl::StencilBuffer::setOp(StencilOp stencilFail, StencilOp stencilZFail, StencilOp stencilZPass) {

    if (currentStencilFail != stencilFail ||
        currentStencilZFail != stencilZFail ||
        currentStencilZPass != stencilZPass) {

        glStencilOp(as_integer(stencilFail), as_integer(stencilZFail), as_integer(stencilZPass));

        currentStencilFail = stencilFail;
        currentStencilZFail = stencilZFail;
        currentStencilZPass = stencilZPass;
    }
}

void gl::StencilBuffer::setLocked(bool lock) {

    locked = lock;
}

void gl::StencilBuffer::setClear(int stencil) {

    if (currentStencilClear != stencil) {

        glClearStencil(stencil);
        currentStencilClear = stencil;
    }
}

void gl::StencilBuffer::reset() {

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

gl::GLState::GLState(): maxTextures(glGetParameteri(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS)) {

    GLint scissorParam[4];
    GLint viewportParam[4];
    glGetIntegerv(GL_SCISSOR_BOX, scissorParam);
    glGetIntegerv(GL_VIEWPORT, viewportParam);

    currentScissor.set((float) scissorParam[0], (float) scissorParam[1], (float) scissorParam[2], (float) scissorParam[3]);
    currentViewport.set((float) viewportParam[0], (float) viewportParam[1], (float) viewportParam[2], (float) viewportParam[3]);

    auto enableLambda = [&](int id) {
        enable(id);
    };

    auto disableLambda = [&](int id) {
        disable(id);
    };

    std::function<GLuint(GLenum, GLenum, int)> createTexture = [](GLenum type, GLenum target, int count) {
        uint8_t data[4];// 4 is required to match default unpack alignment of 4.
        GLuint texture;
        glGenTextures(1, &texture);

        glBindTexture(type, texture);
        glTexParameteri(type, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(type, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        for (int i = 0; i < count; i++) {
            glTexImage2D(target + i, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        }

        return texture;
    };

    depthBuffer.setFunctions(enableLambda, disableLambda);
    stencilBuffer.setFunctions(enableLambda, disableLambda);

    emptyTextures[GL_TEXTURE_2D] = createTexture(GL_TEXTURE_2D, GL_TEXTURE_2D, 1);
    emptyTextures[GL_TEXTURE_CUBE_MAP] = createTexture(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_CUBE_MAP_POSITIVE_X, 6);

    colorBuffer.setClear(0, 0, 0, 1);
    depthBuffer.setClear(1);
    stencilBuffer.setClear(0);

    enable(GL_DEPTH_TEST);
    depthBuffer.setFunc(DepthFunc::LessEqual);

    setFlipSided(false);
    setCullFace(CullFace::Back);
    enable(GL_CULL_FACE);

    setBlending(Blending::Normal);
}

void gl::GLState::enable(int id) {
    if (!enabledCapabilities.contains(id) || enabledCapabilities.at(id) == false) {

        glEnable(id);
        enabledCapabilities[id] = true;
    }
}

void gl::GLState::disable(int id) {
    if (enabledCapabilities.contains(id) && enabledCapabilities.at(id) == true) {

        glDisable(id);
        enabledCapabilities[id] = false;
    }
}

bool gl::GLState::bindFramebuffer(int target, unsigned int framebuffer) {

    if (!currentBoundFramebuffers.contains(target) || (currentBoundFramebuffers.contains(target) && currentBoundFramebuffers.at(target) != framebuffer)) {

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

bool gl::GLState::useProgram(unsigned int program) {

    if (currentProgram != program) {

        glUseProgram(program);

        currentProgram = program;

        return true;
    }

    return false;
}

void gl::GLState::setBlending(
        Blending blending,
        std::optional<BlendEquation> blendEquation,
        std::optional<BlendFactor> blendSrc,
        std::optional<BlendFactor> blendDst,
        std::optional<BlendEquation> blendEquationAlpha,
        std::optional<BlendFactor> blendSrcAlpha,
        std::optional<BlendFactor> blendDstAlpha,
        std::optional<bool> premultipliedAlpha) {

    if (blending == Blending::None) {

        if (currentBlendingEnabled) {

            disable(GL_BLEND);
            currentBlendingEnabled = false;
        }

        return;
    }

    if (!currentBlendingEnabled) {

        enable(GL_BLEND);
        currentBlendingEnabled = true;
    }

    if (blending != Blending::Custom) {

        if (blending != currentBlending || premultipliedAlpha != currentPremultipledAlpha) {

            if (currentBlendEquation != BlendEquation::Add || currentBlendEquationAlpha != BlendEquation::Add) {

                glBlendEquation(GL_FUNC_ADD);

                currentBlendEquation = BlendEquation::Add;
                currentBlendEquationAlpha = BlendEquation::Add;
            }

            if (premultipliedAlpha && premultipliedAlpha.value()) {

                switch (blending) {

                    case Blending::Normal:
                        glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
                        break;

                    case Blending::Additive:
                        glBlendFunc(GL_ONE, GL_ONE);
                        break;

                    case Blending::Subtractive:
                        glBlendFuncSeparate(GL_ZERO, GL_ZERO, GL_ONE_MINUS_SRC_COLOR, GL_ONE_MINUS_SRC_ALPHA);
                        break;

                    case Blending::Multiply:
                        glBlendFuncSeparate(GL_ZERO, GL_SRC_COLOR, GL_ZERO, GL_SRC_ALPHA);
                        break;

                    default:
                        std::cerr << "THREE.GLState: Invalid blending: " << as_integer(blending) << std::endl;
                        break;
                }

            } else {

                switch (blending) {

                    case Blending::Normal:
                        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
                        break;

                    case Blending::Additive:
                        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
                        break;

                    case Blending::Subtractive:
                        glBlendFunc(GL_ZERO, GL_ONE_MINUS_SRC_COLOR);
                        break;

                    case Blending::Multiply:
                        glBlendFunc(GL_ZERO, GL_SRC_COLOR);
                        break;

                    default:
                        std::cerr << "THREE.GLState: Invalid blending: " << as_integer(blending) << std::endl;
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

    std::optional<BlendEquation> blendEquationAlpha_ = blendEquationAlpha;
    if (!blendEquationAlpha_) {
        blendEquationAlpha_ = blendEquation;
    }
    std::optional<BlendFactor> blendSrcAlpha_ = blendSrcAlpha;
    if (!blendSrcAlpha_) {
        blendSrcAlpha_ = blendSrc;
    }
    std::optional<BlendFactor> blendDstAlpha_ = blendDstAlpha;
    if (!blendDstAlpha_) {
        blendDstAlpha_ = blendDst;
    }

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

void gl::GLState::setMaterial(const Material* material, bool frontFaceCW) {

    material->side == Side::Double
            ? disable(GL_CULL_FACE)
            : enable(GL_CULL_FACE);

    auto flipSided = (material->side == Side::Back);
    if (frontFaceCW) flipSided = !flipSided;

    setFlipSided(flipSided);

    (material->blending == Blending::Normal && !material->transparent)
            ? setBlending(Blending::None)
            : setBlending(
                      material->blending,
                      material->blendEquation,
                      material->blendSrc,
                      material->blendDst,
                      material->blendEquationAlpha,
                      material->blendSrcAlpha,
                      material->blendDstAlpha,
                      material->premultipliedAlpha);

    depthBuffer.setFunc(material->depthFunc);
    depthBuffer.setTest(material->depthTest);
    depthBuffer.setMask(material->depthWrite);
    colorBuffer.setMask(material->colorWrite);

    const auto stencilWrite = material->stencilWrite;
    stencilBuffer.setTest(stencilWrite);
    if (stencilWrite) {

        stencilBuffer.setMask(material->stencilWriteMask);
        stencilBuffer.setFunc(material->stencilFunc, material->stencilRef, material->stencilFuncMask);
        stencilBuffer.setOp(material->stencilFail, material->stencilZFail, material->stencilZPass);
    }

    setPolygonOffset(material->polygonOffset, material->polygonOffsetFactor, material->polygonOffsetUnits);

    material->alphaToCoverage ? enable(GL_SAMPLE_ALPHA_TO_COVERAGE) : disable(GL_SAMPLE_ALPHA_TO_COVERAGE);
}

void gl::GLState::setFlipSided(bool flipSided) {

    if (currentFlipSided != flipSided) {

        if (flipSided) {

            glFrontFace(GL_CW);

        } else {

            glFrontFace(GL_CCW);
        }

        currentFlipSided = flipSided;
    }
}

void gl::GLState::setCullFace(CullFace cullFace) {

    if (cullFace != CullFace::None) {

        enable(GL_CULL_FACE);

        if (cullFace != currentCullFace) {

            if (cullFace == CullFace::Back) {

                glCullFace(GL_BACK);

            } else if (cullFace == CullFace::Front) {

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

void gl::GLState::setLineWidth(float width) {

    if (width != currentLineWidth) {

        if (lineWidthAvailable) glLineWidth(width);

        currentLineWidth = width;
    }
}

void gl::GLState::setPolygonOffset(bool polygonOffset, std::optional<float> factor, std::optional<float> units) {

    if (polygonOffset) {

        enable(GL_POLYGON_OFFSET_FILL);

        if (factor && currentPolygonOffsetFactor != *factor || units && currentPolygonOffsetUnits != *units) {

            glPolygonOffset(*factor, *units);

            currentPolygonOffsetFactor = *factor;
            currentPolygonOffsetUnits = *units;
        }

    } else {

        disable(GL_POLYGON_OFFSET_FILL);
    }
}

void gl::GLState::setScissorTest(bool scissorTest) {

    if (scissorTest) {

        enable(GL_SCISSOR_TEST);

    } else {

        disable(GL_SCISSOR_TEST);
    }
}

void gl::GLState::activeTexture(std::optional<GLenum> glSlot) {

    if (!glSlot) glSlot = (GLenum) (GL_TEXTURE0 + maxTextures - 1);

    if (currentTextureSlot != glSlot) {

        glActiveTexture(*glSlot);
        currentTextureSlot = glSlot;
    }
}

void gl::GLState::bindTexture(int glType, std::optional<int> glTexture) {

    if (!currentTextureSlot) {

        activeTexture();
    }

    if (!currentBoundTextures.contains(*currentTextureSlot)) {

        BoundTexture boundTexture{};
        currentBoundTextures[*currentTextureSlot] = boundTexture;
    }

    auto boundTexture = currentBoundTextures.at(*currentTextureSlot);

    if (boundTexture.type != glType || boundTexture.texture != glTexture) {

        glBindTexture(glType, glTexture.value_or(emptyTextures[glType]));

        boundTexture.type = glType;
        boundTexture.texture = glTexture;
    }
}

void gl::GLState::unbindTexture() {

    if (currentTextureSlot && currentBoundTextures.contains(*currentTextureSlot)) {

        auto& boundTexture = currentBoundTextures.at(*currentTextureSlot);

        if (boundTexture.type) {

            glBindTexture(*boundTexture.type, 0);

            boundTexture.type = std::nullopt;
            boundTexture.texture = std::nullopt;
        }
    }
}

void gl::GLState::texImage2D(GLuint target, GLint level, GLint internalFormat, GLint width, GLint height, GLuint format, GLuint type, const void* pixels) {

    glTexImage2D(target, level, internalFormat, width, height, 0, format, type, pixels);
}

void gl::GLState::texImage3D(GLuint target, GLint level, GLint internalFormat, GLint width, GLint height, GLint depth, GLuint format, GLuint type, const void* pixels) {

    glTexImage3D(target, level, internalFormat, width, height, depth, 0, format, type, pixels);
}

void gl::GLState::scissor(const Vector4& scissor) {

    if (!currentScissor.equals(scissor)) {

        glScissor((GLint) scissor.x, (GLint) scissor.y, (GLsizei) scissor.z, (GLsizei) scissor.w);
        currentScissor.copy(scissor);
    }
}

void gl::GLState::viewport(const Vector4& viewport) {

    if (!currentViewport.equals(viewport)) {

        glViewport((GLint) viewport.x, (GLint) viewport.y, (GLsizei) viewport.z, (GLsizei) viewport.w);
        currentViewport.copy(viewport);
    }
}

void gl::GLState::reset(std::pair<int, int> size) {

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

    glScissor(0, 0, size.first, size.second);
    glViewport(0, 0, size.first, size.second);

    // reset internals

    enabledCapabilities.clear();

    currentTextureSlot = std::nullopt;
    currentBoundTextures.clear();

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

    currentScissor.set(0, 0, size.first, size.second);
    currentViewport.set(0, 0, size.first, size.second);

    colorBuffer.reset();
    depthBuffer.reset();
    stencilBuffer.reset();
}
