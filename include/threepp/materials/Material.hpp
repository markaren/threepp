// https://github.com/mrdoob/three.js/blob/r129/src/materials/Material.js

#ifndef THREEPP_MATERIAL_HPP
#define THREEPP_MATERIAL_HPP

#include <threepp/constants.hpp>
#include <threepp/core/EventDispatcher.hpp>
#include <threepp/math/MathUtils.hpp>
#include <threepp/math/Plane.hpp>

#include <optional>

namespace threepp {

    class Material : private EventDispatcher {

    public:
        unsigned int id = materialId++;

        std::string uuid = math::generateUUID();

        bool fog = true;

        int blending = NormalBlending;
        int side = FrontSide;
        bool vertexColors = false;

        float opacity = 1;
        bool transparent = false;

        int blendSrc = SrcAlphaFactor;
        int blendDst = OneMinusSrcAlphaFactor;
        int blendEquation = AddEquation;
        int blendSrcAlpha = -1;
        int blendDstAlpha = -1;
        int blendEquationAlpha = -1;

        int depthFunc = LessEqualDepth;
        bool depthTest = true;
        bool depthWrite = true;

        int stencilWriteMask = 0xff;
        int stencilFunc = AlwaysStencilFunc;
        int stencilRef = 0;
        int stencilFuncMask = 0xff;
        int stencilFail = KeepStencilOp;
        int stencilZFail = KeepStencilOp;
        int stencilZPass = KeepStencilOp;
        bool stencilWrite = false;

        std::vector<Plane> clippingPlanes;
        bool clipIntersection = false;
        bool clipShadows = false;

        std::optional<int> shadowSide;

        bool colorWrite = true;

        bool polygonOffset = false;
        float polygonOffsetFactor = 0;
        float polygonOffsetUnits = 0;

        bool dithering = false;

        float alphaTest = 0;
        bool alphaToCoverage = false;
        bool premultipliedAlpha = false;

        bool visible = true;

        bool toneMapped = true;

        unsigned int version = 0;

        void dispose() {

            dispatchEvent("dispose");
        }

        void needsUpdate() {

            this->version++;
        }

        [[nodiscard]] virtual std::string type() const = 0;

        virtual ~Material() = default;

    protected:
        Material() = default;

    private:
        inline static unsigned int materialId = 0;
    };

}// namespace threepp

#endif//THREEPP_MATERIAL_HPP
