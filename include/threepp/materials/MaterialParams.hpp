
#ifndef THREEPP_MATERIALPARAMS_HPP
#define THREEPP_MATERIALPARAMS_HPP

#include "threepp/materials/Material.hpp"

#include <optional>
#include <string>
#include <utility>

namespace threepp {

    // CRTP base for the fluent, typed material parameter builders. It holds the common
    // Material base-class fields; each concrete material's nested Params derives from
    // MaterialParams<Params> and adds its own fields. Every setter returns Self&, so a
    // chain that mixes base and material-specific setters stays strongly typed and may be
    // written in ANY order (unlike C++20 designated initializers).
    //
    //     auto m = MeshStandardMaterial::create(
    //         MeshStandardMaterial::Params{}
    //             .transparent(true)   // base field (this class)
    //             .roughness(0.4f)     // material-specific field (derived)
    //             .opacity(0.5f));     // base field again -- order-free
    //
    // Only the fields you set are applied; the rest keep their constructor defaults.
    template<class Self>
    class MaterialParams {

    public:
#define TPP_BASE_PARAM(type, field)       \
    Self& field(type v) {                 \
        field##_ = std::move(v);          \
        return static_cast<Self&>(*this); \
    }
        TPP_BASE_PARAM(std::string, name)
        TPP_BASE_PARAM(bool, fog)
        TPP_BASE_PARAM(Blending, blending)
        TPP_BASE_PARAM(Side, side)
        TPP_BASE_PARAM(bool, vertexColors)
        TPP_BASE_PARAM(float, opacity)
        TPP_BASE_PARAM(bool, transparent)
        TPP_BASE_PARAM(BlendFactor, blendSrc)
        TPP_BASE_PARAM(BlendFactor, blendDst)
        TPP_BASE_PARAM(BlendEquation, blendEquation)
        TPP_BASE_PARAM(BlendFactor, blendSrcAlpha)
        TPP_BASE_PARAM(BlendFactor, blendDstAlpha)
        TPP_BASE_PARAM(BlendEquation, blendEquationAlpha)
        TPP_BASE_PARAM(DepthFunc, depthFunc)
        TPP_BASE_PARAM(bool, depthTest)
        TPP_BASE_PARAM(bool, depthWrite)
        TPP_BASE_PARAM(int, stencilWriteMask)
        TPP_BASE_PARAM(StencilFunc, stencilFunc)
        TPP_BASE_PARAM(int, stencilRef)
        TPP_BASE_PARAM(int, stencilFuncMask)
        TPP_BASE_PARAM(StencilOp, stencilFail)
        TPP_BASE_PARAM(StencilOp, stencilZFail)
        TPP_BASE_PARAM(StencilOp, stencilZPass)
        TPP_BASE_PARAM(bool, stencilWrite)
        TPP_BASE_PARAM(Side, shadowSide)
        TPP_BASE_PARAM(bool, colorWrite)
        TPP_BASE_PARAM(bool, polygonOffset)
        TPP_BASE_PARAM(float, polygonOffsetFactor)
        TPP_BASE_PARAM(float, polygonOffsetUnits)
        TPP_BASE_PARAM(bool, dithering)
        TPP_BASE_PARAM(float, alphaTest)
        TPP_BASE_PARAM(bool, alphaToCoverage)
        TPP_BASE_PARAM(bool, premultipliedAlpha)
        TPP_BASE_PARAM(bool, visible)
        TPP_BASE_PARAM(bool, toneMapped)
#undef TPP_BASE_PARAM

        // Apply the base fields that were set onto `m`. Implementation detail used by each
        // material's create(const Params&); not intended to be called directly.
        void applyBaseTo(Material& m) const {
#define TPP_BASE_APPLY(field) \
    if (field##_) m.field = *field##_;
            TPP_BASE_APPLY(name)
            TPP_BASE_APPLY(fog)
            TPP_BASE_APPLY(blending)
            TPP_BASE_APPLY(side)
            TPP_BASE_APPLY(vertexColors)
            TPP_BASE_APPLY(opacity)
            TPP_BASE_APPLY(transparent)
            TPP_BASE_APPLY(blendSrc)
            TPP_BASE_APPLY(blendDst)
            TPP_BASE_APPLY(blendEquation)
            TPP_BASE_APPLY(blendSrcAlpha)
            TPP_BASE_APPLY(blendDstAlpha)
            TPP_BASE_APPLY(blendEquationAlpha)
            TPP_BASE_APPLY(depthFunc)
            TPP_BASE_APPLY(depthTest)
            TPP_BASE_APPLY(depthWrite)
            TPP_BASE_APPLY(stencilWriteMask)
            TPP_BASE_APPLY(stencilFunc)
            TPP_BASE_APPLY(stencilRef)
            TPP_BASE_APPLY(stencilFuncMask)
            TPP_BASE_APPLY(stencilFail)
            TPP_BASE_APPLY(stencilZFail)
            TPP_BASE_APPLY(stencilZPass)
            TPP_BASE_APPLY(stencilWrite)
            TPP_BASE_APPLY(shadowSide)
            TPP_BASE_APPLY(colorWrite)
            TPP_BASE_APPLY(polygonOffset)
            TPP_BASE_APPLY(polygonOffsetFactor)
            TPP_BASE_APPLY(polygonOffsetUnits)
            TPP_BASE_APPLY(dithering)
            TPP_BASE_APPLY(alphaTest)
            TPP_BASE_APPLY(alphaToCoverage)
            TPP_BASE_APPLY(premultipliedAlpha)
            TPP_BASE_APPLY(visible)
            TPP_BASE_APPLY(toneMapped)
#undef TPP_BASE_APPLY
        }

    protected:
        std::optional<std::string> name_;
        std::optional<bool> fog_;
        std::optional<Blending> blending_;
        std::optional<Side> side_;
        std::optional<bool> vertexColors_;
        std::optional<float> opacity_;
        std::optional<bool> transparent_;
        std::optional<BlendFactor> blendSrc_;
        std::optional<BlendFactor> blendDst_;
        std::optional<BlendEquation> blendEquation_;
        std::optional<BlendFactor> blendSrcAlpha_;
        std::optional<BlendFactor> blendDstAlpha_;
        std::optional<BlendEquation> blendEquationAlpha_;
        std::optional<DepthFunc> depthFunc_;
        std::optional<bool> depthTest_;
        std::optional<bool> depthWrite_;
        std::optional<int> stencilWriteMask_;
        std::optional<StencilFunc> stencilFunc_;
        std::optional<int> stencilRef_;
        std::optional<int> stencilFuncMask_;
        std::optional<StencilOp> stencilFail_;
        std::optional<StencilOp> stencilZFail_;
        std::optional<StencilOp> stencilZPass_;
        std::optional<bool> stencilWrite_;
        std::optional<Side> shadowSide_;
        std::optional<bool> colorWrite_;
        std::optional<bool> polygonOffset_;
        std::optional<float> polygonOffsetFactor_;
        std::optional<float> polygonOffsetUnits_;
        std::optional<bool> dithering_;
        std::optional<float> alphaTest_;
        std::optional<bool> alphaToCoverage_;
        std::optional<bool> premultipliedAlpha_;
        std::optional<bool> visible_;
        std::optional<bool> toneMapped_;
    };

}// namespace threepp

#endif//THREEPP_MATERIALPARAMS_HPP
