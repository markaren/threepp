// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLState.js

#ifndef THREEPP_GLSTATE_HPP
#define THREEPP_GLSTATE_HPP

#include "threepp/math/Vector4.hpp"

#include <functional>
#include <optional>
#include <unordered_map>
#include <utility>

namespace threepp {

    class Material;

    namespace gl {

        struct BoundTexture {

            std::optional<int> type;
            std::optional<int> texture;
        };

        struct ColorBuffer {

            bool locked = false;

            Vector4 color{};
            std::optional<bool> currentColorMask;
            Vector4 currentColorClear{0, 0, 0, 0};

            void setMask(bool colorMask);

            void setLocked(bool lock);

            void setClear(float r, float g, float b, float a, bool premultipliedAlpha = false);

            void reset();
        };

        struct DepthBuffer {

            bool locked = false;

            std::optional<bool> currentDepthMask;
            std::optional<int> currentDepthFunc;
            std::optional<float> currentDepthClear;

            std::function<void(int)> enable;
            std::function<void(int)> disable;

            void setFunctions(std::function<void(int)> enable, std::function<void(int)> disable);

            void setTest(bool depthTest);

            void setMask(bool depthMask);

            void setFunc(int depthFunc);

            void setLocked(bool lock);

            void setClear(float depth);

            void reset();
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

            void setFunctions(std::function<void(int)> enable, std::function<void(int)> disable);

            void setTest(bool stencilTest);

            void setMask(int stencilMask);

            void setFunc(int stencilFunc, int stencilRef, int stencilMask);

            void setOp(int stencilFail, int stencilZFail, int stencilZPass);

            void setLocked(bool lock);

            void setClear(int stencil);

            void reset();
        };

        struct GLState {

            ColorBuffer colorBuffer;
            DepthBuffer depthBuffer;
            StencilBuffer stencilBuffer;

            std::unordered_map<int, bool> enabledCapabilities;

            std::unordered_map<int, unsigned int> currentBoundFramebuffers;

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

            const int maxTextures;

            bool lineWidthAvailable = false;
            unsigned int version = 0;

            std::optional<int> currentTextureSlot;
            std::unordered_map<int, BoundTexture> currentBoundTextures;

            std::unordered_map<int, int> emptyTextures;

            Vector4 currentScissor;
            Vector4 currentViewport;

            GLState();

            void enable(int id);

            void disable(int id);

            bool bindFramebuffer(int target, unsigned int framebuffer);

            bool useProgram(unsigned int program, bool force);

            void setBlending(
                    int blending,
                    std::optional<int> blendEquation = std::nullopt,
                    std::optional<int> blendSrc = std::nullopt,
                    std::optional<int> blendDst = std::nullopt,
                    std::optional<int> blendEquationAlpha = std::nullopt,
                    std::optional<int> blendSrcAlpha = std::nullopt,
                    std::optional<int> blendDstAlpha = std::nullopt,
                    std::optional<bool> premultipliedAlpha = std::nullopt);


            void setMaterial(const Material* material, bool frontFaceCW);

            void setFlipSided(bool flipSided);

            void setCullFace(int cullFace);

            void setLineWidth(float width);

            void setPolygonOffset(bool polygonOffset, std::optional<float> factor = std::nullopt, std::optional<float> units = std::nullopt);

            void setScissorTest(bool scissorTest);

            // texture

            void activeTexture(std::optional<unsigned int> glSlot = std::nullopt);

            void bindTexture(int glType, std::optional<int> glTexture);

            void unbindTexture();

            void texImage2D(unsigned int target, int level, int internalFormat, int width, int height, unsigned int format, unsigned int type, const void* pixels);

            void texImage3D(unsigned int target, int level, int internalFormat, int width, int height, int depth, unsigned int format, unsigned int type, const void* pixels);

            //

            void scissor(const Vector4& scissor);

            void viewport(const Vector4& viewport);

            //

            void reset(int width, int height);
        };

    }// namespace gl

}// namespace threepp

#endif//THREEPP_GLSTATE_HPP
