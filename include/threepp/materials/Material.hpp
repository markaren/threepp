// https://github.com/mrdoob/three.js/blob/r129/src/materials/Material.js

#ifndef THREEPP_MATERIAL_HPP
#define THREEPP_MATERIAL_HPP

#include <threepp/constants.hpp>
#include <threepp/math/MathUtils.hpp>
#include <threepp/core/EventDispatcher.hpp>

namespace threepp {

    class Material: public EventDispatcher  {

    public:
        unsigned int id = materialId++;

        std::string uuid = generateUUID();

        std::string name = "";
        std::string type = "Material";

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

//        clippingPlanes = null;
        bool clipIntersection = false;
        bool clipShadows = false;

//        shadowSide = null;

        bool colorWrite = true;

//        precision = null;

        bool polygonOffset = false;
        float polygonOffsetFactor = 0;
        float polygonOffsetUnits = 0;

        bool dithering = false;

        float alphaTest = 0;
        bool alphaToCoverage = false;
        bool premultipliedAlpha = false;

        bool visible = true;

        bool toneMapped = true;

//        std::unordered_map userData;

        unsigned int version = 0;

        void dispose() {

            dispatchEvent("dispose")

        }

        void needsUpdate(bool value) {

            if (value) this->version++;

        }

    private:

        static unsigned int materialId = 0;

    };

}

#endif//THREEPP_MATERIAL_HPP
