// Guards Material::clone() against the recurring "a field was added to an
// interface mixin but the material's copyInto() was never updated" bug — a
// silent one: the clone simply renders without that field, no warning.
//
// Two layers of protection:
//
//   1. A compile-time tripwire. Every mixin in interfaces.hpp is registered
//      below with its full field list, and the macro builds a mirror struct
//      from that list (`decltype(Mixin::field) field;`, same order, same
//      virtual base). If a field is added to or removed from the real mixin
//      without updating the list here, sizeof stops matching and the
//      static_assert fires. Whoever fixes it lands right next to (2).
//
//   2. A run-time round-trip. For every concrete material, each field of every
//      mixin it derives from is set to a distinct non-default value, the
//      material is cloned, and the clone is compared field by field.
//
// So: adding a field to a mixin breaks this build until the field is listed
// here, and listing it here fails the test until copyInto() copies it.

#include "threepp/materials/materials.hpp"
// Not covered by the materials.hpp umbrella.
#include "threepp/materials/MeshDepthMaterial.hpp"
#include "threepp/materials/MeshMatcapMaterial.hpp"
#include "threepp/materials/MeshToonMaterial.hpp"
#include "threepp/materials/RawShaderMaterial.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
// Lives under src/, which the test target puts on the include path.
#include "threepp/materials/MeshDistanceMaterial.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>

using namespace threepp;

namespace {

    // --- MSVC-compatible FOR_EACH (the build runs cl.exe with the traditional
    // preprocessor, hence the TPP_EXPAND indirections). ------------------------

#define TPP_EXPAND(x) x

#define TPP_FE_1(W, a) W(a)
#define TPP_FE_2(W, a, ...) W(a) TPP_EXPAND(TPP_FE_1(W, __VA_ARGS__))
#define TPP_FE_3(W, a, ...) W(a) TPP_EXPAND(TPP_FE_2(W, __VA_ARGS__))
#define TPP_FE_4(W, a, ...) W(a) TPP_EXPAND(TPP_FE_3(W, __VA_ARGS__))
#define TPP_FE_5(W, a, ...) W(a) TPP_EXPAND(TPP_FE_4(W, __VA_ARGS__))
#define TPP_FE_6(W, a, ...) W(a) TPP_EXPAND(TPP_FE_5(W, __VA_ARGS__))
#define TPP_FE_7(W, a, ...) W(a) TPP_EXPAND(TPP_FE_6(W, __VA_ARGS__))
#define TPP_FE_8(W, a, ...) W(a) TPP_EXPAND(TPP_FE_7(W, __VA_ARGS__))

#define TPP_FE_NTH(_1, _2, _3, _4, _5, _6, _7, _8, N, ...) N
#define TPP_FOR_EACH(W, ...)                                                  \
    TPP_EXPAND(TPP_FE_NTH(__VA_ARGS__, TPP_FE_8, TPP_FE_7, TPP_FE_6,          \
                          TPP_FE_5, TPP_FE_4, TPP_FE_3, TPP_FE_2, TPP_FE_1)   \
                   (W, __VA_ARGS__))

    // --- Per-field mutate / compare ------------------------------------------

    std::shared_ptr<Texture> freshTexture() {

        return Texture::create();
    }

    void mutateField(float& v, int seed) { v = 1.5f + 0.125f * static_cast<float>(seed); }
    void mutateField(bool& v, int) { v = !v; }
    void mutateField(Color& v, int seed) { v.setRGB(0.1f * static_cast<float>(seed % 7), 0.25f, 0.75f); }
    void mutateField(Vector2& v, int seed) { v.set(2.f + static_cast<float>(seed), 3.f); }
    void mutateField(std::shared_ptr<Texture>& v, int) { v = freshTexture(); }
    void mutateField(NormalMapType& v, int) { v = NormalMapType::ObjectSpace; }
    void mutateField(CombineOperation& v, int) { v = CombineOperation::Add; }
    void mutateField(DepthPacking& v, int) { v = DepthPacking::RGBA; }
    void mutateField(std::optional<Color>& v, int seed) { v = Color(0.2f, 0.4f, 0.1f * static_cast<float>(seed % 5)); }
    void mutateField(std::unordered_map<std::string, std::string>& v, int seed) {

        v["TPP_CLONE_TEST"] = std::to_string(seed);
    }

    // shared_ptr fields are shared by clone(), not deep-copied: compare identity.
    bool sameField(const std::shared_ptr<Texture>& a, const std::shared_ptr<Texture>& b) { return a.get() == b.get(); }

    template<class T>
    bool sameField(const T& a, const T& b) {

        return a == b;
    }

    // --- Mixin registration ---------------------------------------------------

#define TPP_MIRROR_FIELD(f) decltype(MixinUnderTest::f) f;
#define TPP_MUTATE_FIELD(f) mutateField(m.f, seed++);
#define TPP_CHECK_FIELD(f) \
    INFO(#f);              \
    CHECK(sameField(a.f, b.f));

#define TPP_MIXIN_BODY(MIXIN, ...)                                                    \
    void mutateMixin(MIXIN& m, int& seed) {                                           \
        TPP_FOR_EACH(TPP_MUTATE_FIELD, __VA_ARGS__)                                   \
    }                                                                                 \
    void checkMixin(const MIXIN& a, const MIXIN& b) {                                 \
        TPP_FOR_EACH(TPP_CHECK_FIELD, __VA_ARGS__)                                    \
    }

    // Mixins whose only base is `virtual Material` — the common case, mirrored
    // automatically for the size tripwire.
#define TPP_MIXIN(MIXIN, ...)                                                         \
    namespace mirror_##MIXIN {                                                        \
        using MixinUnderTest = MIXIN;                                                 \
        struct Mirror: virtual Material {                                             \
            TPP_FOR_EACH(TPP_MIRROR_FIELD, __VA_ARGS__)                               \
        };                                                                            \
        static_assert(sizeof(Mirror) == sizeof(MIXIN),                                \
                      #MIXIN " gained or lost a field. Add/remove it in the "          \
                             "TPP_MIXIN(" #MIXIN ", ...) list below AND make sure "    \
                             "every copyInto() that owns this mixin copies it.");      \
    }                                                                                 \
    TPP_MIXIN_BODY(MIXIN, __VA_ARGS__)

    TPP_MIXIN(MaterialWithColor, color)
    TPP_MIXIN(MaterialWithRotation, rotation)
    TPP_MIXIN(MaterialWithClipping, clipping)
    TPP_MIXIN(MaterialWithLights, lights)
    TPP_MIXIN(MaterialWithSize, size, sizeAttenuation)
    TPP_MIXIN(MaterialWithLineWidth, linewidth)
    TPP_MIXIN(MaterialWithEmissive, emissive, emissiveIntensity, emissiveMap)
    TPP_MIXIN(MaterialWithSpecular, specular, shininess)
    TPP_MIXIN(MaterialWithRefractionRatio, refractionRatio)
    TPP_MIXIN(MaterialWithWireframe, wireframe, wireframeLinewidth)
    TPP_MIXIN(MaterialWithMap, map)
    TPP_MIXIN(MaterialWithAlphaMap, alphaMap)
    TPP_MIXIN(MaterialWithSpecularMap, specularMap)
    TPP_MIXIN(MaterialWithEnvMap, envMapIntensity, envMap)
    TPP_MIXIN(MaterialWithGradientMap, gradientMap)
    TPP_MIXIN(MaterialWithAoMap, aoMap, aoMapIntensity)
    TPP_MIXIN(MaterialWithBumpMap, bumpMap, bumpScale)
    TPP_MIXIN(MaterialWithLightMap, lightMap, lightMapIntensity)
    TPP_MIXIN(MaterialWithDisplacementMap, displacementMap, displacementScale, displacementBias)
    TPP_MIXIN(MaterialWithNormalMap, normalMap, normalMapType, normalScale)
    TPP_MIXIN(MaterialWithDetailMap, detailMap, detailRepeat, detailStrength, detailNormalMap, detailNormalScale, detailRoughStrength)
    TPP_MIXIN(MaterialWithTranslucency, translucency, translucencyColor)
    TPP_MIXIN(MaterialWithMatCap, matcap)
    TPP_MIXIN(MaterialWithRoughness, roughness, roughnessMap)
    TPP_MIXIN(MaterialWithMetalness, metalness, metalnessMap)
    TPP_MIXIN(MaterialWithThickness, thickness, thicknessMap, thinWalled)
    TPP_MIXIN(MaterialWithClearcoat, clearcoat, clearcoatMap, clearcoatRoughness, clearcoatRoughnessMap, clearcoatNormalScale, clearcoatNormalMap)
    TPP_MIXIN(MaterialWithTransmission, transmission, ior, dispersion, transmissionMap)
    TPP_MIXIN(MaterialWithAttenuation, attenuationDistance, attenuationColor, scatterDistance, scatterColor)
    TPP_MIXIN(MaterialWithSheen, sheenColor, sheenRoughness)
    TPP_MIXIN(MaterialWithIridescence, iridescence, iridescenceIOR, iridescenceThicknessNm)
    TPP_MIXIN(MaterialWithPbrSpecular, specularIntensity, specularColor)
    TPP_MIXIN(MaterialWithCombine, combine)
    TPP_MIXIN(MaterialWithDepthPacking, depthPacking)
    TPP_MIXIN(MaterialWithFlatShading, flatShading)
    TPP_MIXIN(MaterialWithVertexTangents, vertexTangents)
    TPP_MIXIN(MaterialWithDefines, defines)
    TPP_MIXIN(MaterialWithMorphTargets, morphTargets, morphNormals)

    // MaterialWithReflectivity is the one mixin with a second base, so its
    // mirror is spelled out rather than generated.
    namespace mirror_MaterialWithReflectivity {
        struct Mirror: virtual Material, virtual MaterialWithRefractionRatio {
            float reflectivity;
        };
        static_assert(sizeof(Mirror) == sizeof(MaterialWithReflectivity),
                      "MaterialWithReflectivity gained or lost a field. Update the list below "
                      "AND every copyInto() that owns this mixin.");
    }// namespace mirror_MaterialWithReflectivity
    TPP_MIXIN_BODY(MaterialWithReflectivity, reflectivity)

    // Every mixin, for the dispatch below. Keep in sync with the list above.
#define TPP_ALL_MIXINS(X)              \
    X(MaterialWithColor)               \
    X(MaterialWithRotation)            \
    X(MaterialWithClipping)            \
    X(MaterialWithLights)              \
    X(MaterialWithSize)                \
    X(MaterialWithLineWidth)           \
    X(MaterialWithEmissive)            \
    X(MaterialWithSpecular)            \
    X(MaterialWithRefractionRatio)     \
    X(MaterialWithReflectivity)        \
    X(MaterialWithWireframe)           \
    X(MaterialWithMap)                 \
    X(MaterialWithAlphaMap)            \
    X(MaterialWithSpecularMap)         \
    X(MaterialWithEnvMap)              \
    X(MaterialWithGradientMap)         \
    X(MaterialWithAoMap)               \
    X(MaterialWithBumpMap)             \
    X(MaterialWithLightMap)            \
    X(MaterialWithDisplacementMap)     \
    X(MaterialWithNormalMap)           \
    X(MaterialWithDetailMap)           \
    X(MaterialWithTranslucency)        \
    X(MaterialWithMatCap)              \
    X(MaterialWithRoughness)           \
    X(MaterialWithMetalness)           \
    X(MaterialWithThickness)           \
    X(MaterialWithClearcoat)           \
    X(MaterialWithTransmission)        \
    X(MaterialWithAttenuation)         \
    X(MaterialWithSheen)               \
    X(MaterialWithIridescence)         \
    X(MaterialWithPbrSpecular)         \
    X(MaterialWithCombine)             \
    X(MaterialWithDepthPacking)        \
    X(MaterialWithFlatShading)         \
    X(MaterialWithVertexTangents)      \
    X(MaterialWithDefines)             \
    X(MaterialWithMorphTargets)

    // `is_convertible` rather than `is_base_of`: it is false both for "not a
    // base" and for "ambiguous base", so a mixin reachable by two paths is
    // skipped rather than failing to compile.
    template<class MIXIN, class Mat>
    void mutateOne(Mat& m, int& seed) {

        if constexpr (std::is_convertible_v<Mat*, MIXIN*>) {
            mutateMixin(static_cast<MIXIN&>(m), seed);
        }
    }

    template<class MIXIN, class Mat>
    void checkOne(const Mat& a, const Mat& b) {

        if constexpr (std::is_convertible_v<const Mat*, const MIXIN*>) {
            checkMixin(static_cast<const MIXIN&>(a), static_cast<const MIXIN&>(b));
        }
    }

    template<class Mat>
    void mutateMixins(Mat& m) {

        int seed = 1;
#define TPP_MUTATE_ONE(MIXIN) mutateOne<MIXIN>(m, seed);
        TPP_ALL_MIXINS(TPP_MUTATE_ONE)
#undef TPP_MUTATE_ONE
    }

    template<class Mat>
    void checkMixins(const Mat& a, const Mat& b) {

#define TPP_CHECK_ONE(MIXIN) checkOne<MIXIN>(a, b);
        TPP_ALL_MIXINS(TPP_CHECK_ONE)
#undef TPP_CHECK_ONE
    }

    // --- Base Material fields (not a mixin, but the same bug class) ------------

    void mutateBase(Material& m) {

        m.name = "cloned-material";
        m.fog = !m.fog;

        m.blending = Blending::Additive;
        m.side = Side::Double;
        m.vertexColors = true;
        m.textureAnimatedHint = true;

        m.tetSkinning = true;
        m.tetTexture = freshTexture();
        m.tetTextureSize = 64;

        m.opacity = 0.375f;
        m.transparent = true;

        m.blendSrc = BlendFactor::DstColor;
        m.blendDst = BlendFactor::OneMinusDstAlpha;
        m.blendEquation = BlendEquation::Subtract;
        m.blendSrcAlpha = BlendFactor::One;
        m.blendDstAlpha = BlendFactor::Zero;
        m.blendEquationAlpha = BlendEquation::ReverseSubtract;

        m.depthFunc = DepthFunc::Greater;
        m.depthTest = false;
        m.depthWrite = false;

        m.stencilWriteMask = 0x0f;
        m.stencilFunc = StencilFunc::Equal;
        m.stencilRef = 3;
        m.stencilFuncMask = 0x3f;
        m.stencilFail = StencilOp::Replace;
        m.stencilZFail = StencilOp::Invert;
        m.stencilZPass = StencilOp::Increment;
        m.stencilWrite = true;

        m.clippingPlanes = {Plane(Vector3(0, 1, 0), 2.5f)};
        m.clipIntersection = true;
        m.clipShadows = true;
        m.clipping = true;

        m.shadowSide = Side::Back;
        m.colorWrite = false;

        m.polygonOffset = true;
        m.polygonOffsetFactor = 1.25f;
        m.polygonOffsetUnits = 2.5f;

        m.dithering = true;

        m.alphaTest = 0.625f;
        m.alphaToCoverage = true;
        m.premultipliedAlpha = true;

        m.visible = false;
        m.toneMapped = false;

        m.defaultAttributeValues["tppCloneTest"] = Vector2(1, 2);
    }

    void checkBase(const Material& a, const Material& b) {

        CHECK(a.name == b.name);
        CHECK(a.fog == b.fog);

        CHECK(a.blending == b.blending);
        CHECK(a.side == b.side);
        CHECK(a.vertexColors == b.vertexColors);
        CHECK(a.textureAnimatedHint == b.textureAnimatedHint);

        CHECK(a.tetSkinning == b.tetSkinning);
        CHECK(a.tetTexture.get() == b.tetTexture.get());
        CHECK(a.tetTextureSize == b.tetTextureSize);

        CHECK(a.opacity == b.opacity);
        CHECK(a.transparent == b.transparent);

        CHECK(a.blendSrc == b.blendSrc);
        CHECK(a.blendDst == b.blendDst);
        CHECK(a.blendEquation == b.blendEquation);
        CHECK(a.blendSrcAlpha == b.blendSrcAlpha);
        CHECK(a.blendDstAlpha == b.blendDstAlpha);
        CHECK(a.blendEquationAlpha == b.blendEquationAlpha);

        CHECK(a.depthFunc == b.depthFunc);
        CHECK(a.depthTest == b.depthTest);
        CHECK(a.depthWrite == b.depthWrite);

        CHECK(a.stencilWriteMask == b.stencilWriteMask);
        CHECK(a.stencilFunc == b.stencilFunc);
        CHECK(a.stencilRef == b.stencilRef);
        CHECK(a.stencilFuncMask == b.stencilFuncMask);
        CHECK(a.stencilFail == b.stencilFail);
        CHECK(a.stencilZFail == b.stencilZFail);
        CHECK(a.stencilZPass == b.stencilZPass);
        CHECK(a.stencilWrite == b.stencilWrite);

        REQUIRE(a.clippingPlanes.size() == b.clippingPlanes.size());
        for (std::size_t i = 0; i < a.clippingPlanes.size(); ++i) {
            CHECK(a.clippingPlanes[i].constant == b.clippingPlanes[i].constant);
            CHECK(a.clippingPlanes[i].normal.equals(b.clippingPlanes[i].normal));
        }
        CHECK(a.clipIntersection == b.clipIntersection);
        CHECK(a.clipShadows == b.clipShadows);
        CHECK(a.clipping == b.clipping);

        CHECK(a.shadowSide == b.shadowSide);
        CHECK(a.colorWrite == b.colorWrite);

        CHECK(a.polygonOffset == b.polygonOffset);
        CHECK(a.polygonOffsetFactor == b.polygonOffsetFactor);
        CHECK(a.polygonOffsetUnits == b.polygonOffsetUnits);

        CHECK(a.dithering == b.dithering);

        CHECK(a.alphaTest == b.alphaTest);
        CHECK(a.alphaToCoverage == b.alphaToCoverage);
        CHECK(a.premultipliedAlpha == b.premultipliedAlpha);

        CHECK(a.visible == b.visible);
        CHECK(a.toneMapped == b.toneMapped);

        CHECK(a.defaultAttributeValues.size() == b.defaultAttributeValues.size());
        CHECK(b.defaultAttributeValues.count("tppCloneTest") == 1);
    }

    // --- Class-specific fields that live outside any mixin ---------------------

    template<class Mat>
    void mutateExtras(Mat&) {}

    template<class Mat>
    void checkExtras(const Mat&, const Mat&) {}

    template<>
    void mutateExtras<LineDashedMaterial>(LineDashedMaterial& m) {

        m.dashSize = 4.f;
        m.gapSize = 5.f;
        m.scale = 6.f;
    }

    template<>
    void checkExtras<LineDashedMaterial>(const LineDashedMaterial& a, const LineDashedMaterial& b) {

        CHECK(a.dashSize == b.dashSize);
        CHECK(a.gapSize == b.gapSize);
        CHECK(a.scale == b.scale);
    }

    template<>
    void mutateExtras<MeshDistanceMaterial>(MeshDistanceMaterial& m) {

        m.referencePosition.set(1, 2, 3);
        m.nearDistance = 0.25f;
        m.farDistance = 500.f;
    }

    template<>
    void checkExtras<MeshDistanceMaterial>(const MeshDistanceMaterial& a, const MeshDistanceMaterial& b) {

        CHECK(a.referencePosition.equals(b.referencePosition));
        CHECK(a.nearDistance == b.nearDistance);
        CHECK(a.farDistance == b.farDistance);
    }

    void mutateShader(ShaderMaterial& m) {

        m.vertexShader = "// tpp clone test vertex";
        m.fragmentShader = "// tpp clone test fragment";
        m.uniforms.emplace("tppCloneTest", Uniform(UniformValue(1.5f)));
        m.customTextures["tppCloneTest"] = reinterpret_cast<void*>(static_cast<std::uintptr_t>(0xABCD));
        m.index0AttributeName = "position";
        m.uniformsNeedUpdate = true;
    }

    void checkShader(const ShaderMaterial& a, const ShaderMaterial& b) {

        CHECK(a.vertexShader == b.vertexShader);
        CHECK(a.fragmentShader == b.fragmentShader);
        CHECK(a.uniforms.size() == b.uniforms.size());
        CHECK(b.uniforms.count("tppCloneTest") == 1);
        CHECK(a.customTextures == b.customTextures);
        CHECK(a.index0AttributeName == b.index0AttributeName);
        CHECK(a.uniformsNeedUpdate == b.uniformsNeedUpdate);
    }

    template<>
    void mutateExtras<ShaderMaterial>(ShaderMaterial& m) { mutateShader(m); }
    template<>
    void checkExtras<ShaderMaterial>(const ShaderMaterial& a, const ShaderMaterial& b) { checkShader(a, b); }

    template<>
    void mutateExtras<RawShaderMaterial>(RawShaderMaterial& m) { mutateShader(m); }
    template<>
    void checkExtras<RawShaderMaterial>(const RawShaderMaterial& a, const RawShaderMaterial& b) { checkShader(a, b); }

    // --- The round-trip -------------------------------------------------------

    template<class Mat>
    void assertCloneRoundTrips() {

        auto src = Mat::create();
        REQUIRE(src);

        INFO("material: " << src->type());

        mutateBase(*src);
        mutateMixins(*src);
        mutateExtras(*src);

        auto dst = src->template clone<Mat>();
        REQUIRE(dst);
        REQUIRE(dst.get() != src.get());
        CHECK(dst->type() == src->type());

        checkBase(*src, *dst);
        checkMixins(*src, *dst);
        checkExtras(*src, *dst);
    }

}// namespace

TEST_CASE("clone() round-trips every mixin field", "[materials]") {

    assertCloneRoundTrips<LineBasicMaterial>();
    assertCloneRoundTrips<LineDashedMaterial>();
    assertCloneRoundTrips<MeshBasicMaterial>();
    assertCloneRoundTrips<MeshDepthMaterial>();
    assertCloneRoundTrips<MeshDistanceMaterial>();
    assertCloneRoundTrips<MeshLambertMaterial>();
    assertCloneRoundTrips<MeshMatcapMaterial>();
    assertCloneRoundTrips<MeshNormalMaterial>();
    assertCloneRoundTrips<MeshPhongMaterial>();
    assertCloneRoundTrips<MeshPhysicalMaterial>();
    assertCloneRoundTrips<MeshStandardMaterial>();
    assertCloneRoundTrips<MeshToonMaterial>();
    assertCloneRoundTrips<PointsMaterial>();
    assertCloneRoundTrips<RawShaderMaterial>();
    assertCloneRoundTrips<ShaderMaterial>();
    assertCloneRoundTrips<ShadowMaterial>();
    assertCloneRoundTrips<SpriteMaterial>();
}

TEST_CASE("MeshStandardMaterial::clone keeps the whole detail layer", "[materials]") {

    // Regression: detailNormalMap / detailNormalScale / detailRoughStrength were
    // declared on MaterialWithDetailMap but never copied, so clone() silently
    // dropped the detail-normal/roughness layer.
    auto src = MeshStandardMaterial::create();
    src->detailMap = Texture::create();
    src->detailRepeat = 3.5f;
    src->detailStrength = 0.75f;
    src->detailNormalMap = Texture::create();
    src->detailNormalScale = 0.4f;
    src->detailRoughStrength = 0.2f;

    auto dst = src->clone<MeshStandardMaterial>();

    REQUIRE(dst);
    CHECK(dst->detailMap == src->detailMap);
    CHECK(dst->detailRepeat == src->detailRepeat);
    CHECK(dst->detailStrength == src->detailStrength);
    CHECK(dst->detailNormalMap == src->detailNormalMap);
    CHECK(dst->detailNormalScale == src->detailNormalScale);
    CHECK(dst->detailRoughStrength == src->detailRoughStrength);
}

TEST_CASE("MeshPhysicalMaterial has exactly one refractionRatio", "[materials]") {

    // MaterialWithRefractionRatio is reached both via MeshStandardMaterial and
    // via MaterialWithReflectivity. It is a *virtual* base so those collapse to
    // one subobject; if that ever regresses to two, the plain assignment below
    // stops compiling and the dynamic_cast starts returning null.
    auto m = MeshPhysicalMaterial::create();
    m->refractionRatio = 0.5f;

    auto* viaMixin = dynamic_cast<MaterialWithRefractionRatio*>(m.get());
    REQUIRE(viaMixin != nullptr);// null when the base is ambiguous
    CHECK(viaMixin->refractionRatio == 0.5f);

    // The path ObjectLoader/ObjectExporter and GLMaterials each take.
    CHECK(m->as<MaterialWithReflectivity>()->refractionRatio == 0.5f);
    CHECK(static_cast<MeshStandardMaterial*>(m.get())->refractionRatio == 0.5f);

    // ...and the path setValues() takes.
    m->setValues({{"refractionRatio", 0.25f}});
    CHECK(viaMixin->refractionRatio == 0.25f);
}

TEST_CASE("copyCompatibleFrom carries every shared mixin", "[materials]") {

    // The other copy path: a hand-maintained mixin table in Material.cpp that
    // drifted the same way copyInto did.
    auto src = MeshPhysicalMaterial::create();
    mutateBase(*src);
    mutateMixins(*src);

    auto dst = MeshPhysicalMaterial::create();
    dst->copyCompatibleFrom(*src);

    checkBase(*src, *dst);
    checkMixins(*src, *dst);
}
