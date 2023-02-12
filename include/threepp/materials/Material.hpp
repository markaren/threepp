// https://github.com/mrdoob/three.js/blob/r129/src/materials/Material.js

#ifndef THREEPP_MATERIAL_HPP
#define THREEPP_MATERIAL_HPP

#include <threepp/constants.hpp>
#include <threepp/core/EventDispatcher.hpp>
#include <threepp/core/Uniform.hpp>
#include <threepp/math/Plane.hpp>

#include <optional>

namespace threepp {

    class Material : public EventDispatcher {

    public:
        const unsigned int id = materialId++;

        const std::string uuid = utils::generateUUID();

        std::string name;

        bool fog = true;

        int blending = NormalBlending;
        int side = FrontSide;
        bool vertexColors = false;

        float opacity = 1;
        bool transparent = false;

        int blendSrc = SrcAlphaFactor;
        int blendDst = OneMinusSrcAlphaFactor;
        int blendEquation = AddEquation;
        std::optional<int> blendSrcAlpha;
        std::optional<int> blendDstAlpha;
        std::optional<int> blendEquationAlpha;

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

        std::vector<Plane> clippingPlanes{};
        bool clipIntersection = false;
        bool clipShadows = false;
        bool clipping = false;

        std::optional<int> shadowSide{};

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

        std::unordered_map<std::string, UniformValue> defaultAttributeValues;

        unsigned int version = 0;

        Material(const Material&) = delete;

        void dispose() {

            dispatchEvent("dispose", this);
        }

        void needsUpdate() {

            this->version++;
        }

        [[nodiscard]] virtual std::string type() const = 0;

        template<class T>
        T *as() {

            return dynamic_cast<T *>(this);
        }

        template<class T>
        bool is() {

            return dynamic_cast<T *>(this) != nullptr;
        }

        virtual ~Material() = default;

    protected:
        Material() = default;

        void copyInto(Material *m) const {

            m->name = name;

            m->fog = fog;

            m->blending = blending;
            m->side = side;
            m->vertexColors = vertexColors;

            m->opacity = opacity;
            m->transparent = transparent;

            m->blendSrc = blendSrc;
            m->blendDst = blendDst;
            m->blendEquation = blendEquation;
            m->blendSrcAlpha = blendSrcAlpha;
            m->blendDstAlpha = blendDstAlpha;
            m->blendEquationAlpha = blendEquationAlpha;

            m->depthFunc = depthFunc;
            m->depthTest = depthTest;
            m->depthWrite = depthWrite;

            m->stencilWriteMask = stencilWriteMask;
            m->stencilFunc = stencilFunc;
            m->stencilRef = stencilRef;
            m->stencilFuncMask = stencilFuncMask;
            m->stencilFail = stencilFail;
            m->stencilZFail = stencilZFail;
            m->stencilZPass = stencilZPass;
            m->stencilWrite = stencilWrite;

            const auto &srcPlanes = clippingPlanes;
            std::vector<Plane> dstPlanes;

            if (!srcPlanes.empty()) {

                auto n = srcPlanes.size();
                dstPlanes.resize(n);

                for (unsigned i = 0; i != n; ++i) {

                    dstPlanes[i] = srcPlanes[i];
                }
            }

            m->clippingPlanes = dstPlanes;
            m->clipIntersection = clipIntersection;
            m->clipShadows = clipShadows;

            m->shadowSide = shadowSide;

            m->colorWrite = colorWrite;

            m->polygonOffset = polygonOffset;
            m->polygonOffsetFactor = polygonOffsetFactor;
            m->polygonOffsetUnits = polygonOffsetUnits;

            m->dithering = dithering;

            m->alphaTest = alphaTest;
            m->alphaToCoverage = alphaToCoverage;
            m->premultipliedAlpha = premultipliedAlpha;

            m->visible = visible;

            m->toneMapped = toneMapped;
        }

    private:
        inline static unsigned int materialId = 0;
    };

}// namespace threepp

#endif//THREEPP_MATERIAL_HPP
