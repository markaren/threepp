// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLState.js

#ifndef THREEPP_GLSTATE_HPP
#define THREEPP_GLSTATE_HPP

#include "threepp/constants.hpp"
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
            std::optional<DepthFunc> currentDepthFunc;
            std::optional<float> currentDepthClear;

            std::function<void(int)> enable;
            std::function<void(int)> disable;

            void setFunctions(std::function<void(int)> enable, std::function<void(int)> disable);

            void setTest(bool depthTest);

            void setMask(bool depthMask);

            void setFunc(DepthFunc depthFunc);

            void setLocked(bool lock);

            void setClear(float depth);

            void reset();
        };

        struct StencilBuffer {

            bool locked = false;

            std::optional<int> currentStencilMask;
            std::optional<StencilFunc> currentStencilFunc;
            std::optional<int> currentStencilRef;
            std::optional<int> currentStencilFuncMask;
            std::optional<StencilOp> currentStencilFail;
            std::optional<StencilOp> currentStencilZFail;
            std::optional<StencilOp> currentStencilZPass;
            std::optional<int> currentStencilClear;

            std::function<void(int)> enable;
            std::function<void(int)> disable;

            void setFunctions(std::function<void(int)> enable, std::function<void(int)> disable);

            void setTest(bool stencilTest);

            void setMask(int stencilMask);

            void setFunc(StencilFunc stencilFunc, int stencilRef, int stencilMask);

            void setOp(StencilOp stencilFail, StencilOp stencilZFail, StencilOp stencilZPass);

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
            std::optional<Blending> currentBlending;
            std::optional<BlendEquation> currentBlendEquation;
            std::optional<BlendFactor> currentBlendSrc;
            std::optional<BlendFactor> currentBlendDst;
            std::optional<BlendEquation> currentBlendEquationAlpha;
            std::optional<BlendFactor> currentBlendSrcAlpha;
            std::optional<BlendFactor> currentBlendDstAlpha;
            std::optional<bool> currentPremultipledAlpha = false;

            std::optional<bool> currentFlipSided;
            std::optional<CullFace> currentCullFace;

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

            bool useProgram(unsigned int program);

            void setBlending(
                    Blending blending,
                    std::optional<BlendEquation> blendEquation = std::nullopt,
                    std::optional<BlendFactor> blendSrc = std::nullopt,
                    std::optional<BlendFactor> blendDst = std::nullopt,
                    std::optional<BlendEquation> blendEquationAlpha = std::nullopt,
                    std::optional<BlendFactor> blendSrcAlpha = std::nullopt,
                    std::optional<BlendFactor> blendDstAlpha = std::nullopt,
                    std::optional<bool> premultipliedAlpha = std::nullopt);


            void setMaterial(const Material* material, bool frontFaceCW);

            void setFlipSided(bool flipSided);

            void setCullFace(CullFace cullFace);

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
