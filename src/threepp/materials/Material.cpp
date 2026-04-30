
#include "threepp/materials/Material.hpp"

#include "threepp/materials/interfaces.hpp"
#include "threepp/math/MathUtils.hpp"

#include <iostream>
#include <sstream>

using namespace threepp;


Material::Material()
    : uuid_(math::generateUUID()) {}


std::string Material::uuid() const {

    return uuid_;
}

unsigned int Material::version() const {

    return version_;
}

void Material::dispose() {
    if (!disposed_) {
        disposed_ = true;
        dispatchEvent("dispose", this);
    }
}

void Material::needsUpdate() {

    this->version_++;
}

void Material::copyInto(Material& m) const {

    m.name = name;

    m.fog = fog;

    m.blending = blending;
    m.side = side;
    m.vertexColors = vertexColors;

    m.opacity = opacity;
    m.transparent = transparent;

    m.blendSrc = blendSrc;
    m.blendDst = blendDst;
    m.blendEquation = blendEquation;
    m.blendSrcAlpha = blendSrcAlpha;
    m.blendDstAlpha = blendDstAlpha;
    m.blendEquationAlpha = blendEquationAlpha;

    m.depthFunc = depthFunc;
    m.depthTest = depthTest;
    m.depthWrite = depthWrite;

    m.stencilWriteMask = stencilWriteMask;
    m.stencilFunc = stencilFunc;
    m.stencilRef = stencilRef;
    m.stencilFuncMask = stencilFuncMask;
    m.stencilFail = stencilFail;
    m.stencilZFail = stencilZFail;
    m.stencilZPass = stencilZPass;
    m.stencilWrite = stencilWrite;

    const auto& srcPlanes = clippingPlanes;
    std::vector<Plane> dstPlanes;

    if (!srcPlanes.empty()) {

        auto n = srcPlanes.size();
        dstPlanes.resize(n);

        for (unsigned i = 0; i != n; ++i) {

            dstPlanes[i] = srcPlanes[i];
        }
    }

    m.clippingPlanes = dstPlanes;
    m.clipIntersection = clipIntersection;
    m.clipShadows = clipShadows;

    m.shadowSide = shadowSide;

    m.colorWrite = colorWrite;

    m.polygonOffset = polygonOffset;
    m.polygonOffsetFactor = polygonOffsetFactor;
    m.polygonOffsetUnits = polygonOffsetUnits;

    m.dithering = dithering;

    m.alphaTest = alphaTest;
    m.alphaToCoverage = alphaToCoverage;
    m.premultipliedAlpha = premultipliedAlpha;

    m.visible = visible;

    m.toneMapped = toneMapped;
}

void Material::copyCompatibleFrom(const Material& other) {

    // Base Material fields are present on every subclass — copy them via
    // the base copyInto explicitly to bypass virtual dispatch (a sibling
    // override would assume same-type and crash via as<T>()).
    other.Material::copyInto(*this);

    auto& dst = *this;

#define TPP_COPY_MIXIN(MIXIN, BODY)                                    \
    do {                                                               \
        const auto* s = dynamic_cast<const MIXIN*>(&other);            \
        auto* d = dynamic_cast<MIXIN*>(&dst);                          \
        if (s && d) { BODY }                                           \
    } while (0)

    TPP_COPY_MIXIN(MaterialWithColor, d->color = s->color;);
    TPP_COPY_MIXIN(MaterialWithRotation, d->rotation = s->rotation;);
    TPP_COPY_MIXIN(MaterialWithClipping, d->clipping = s->clipping;);
    TPP_COPY_MIXIN(MaterialWithLights, d->lights = s->lights;);
    TPP_COPY_MIXIN(MaterialWithSize,
                   d->size = s->size;
                   d->sizeAttenuation = s->sizeAttenuation;);
    TPP_COPY_MIXIN(MaterialWithLineWidth, d->linewidth = s->linewidth;);
    TPP_COPY_MIXIN(MaterialWithEmissive,
                   d->emissive = s->emissive;
                   d->emissiveIntensity = s->emissiveIntensity;
                   d->emissiveMap = s->emissiveMap;);
    TPP_COPY_MIXIN(MaterialWithSpecular,
                   d->specular = s->specular;
                   d->shininess = s->shininess;);
    TPP_COPY_MIXIN(MaterialWithRefractionRatio, d->refractionRatio = s->refractionRatio;);
    TPP_COPY_MIXIN(MaterialWithReflectivity, d->reflectivity = s->reflectivity;);
    TPP_COPY_MIXIN(MaterialWithWireframe,
                   d->wireframe = s->wireframe;
                   d->wireframeLinewidth = s->wireframeLinewidth;);
    TPP_COPY_MIXIN(MaterialWithMap, d->map = s->map;);
    TPP_COPY_MIXIN(MaterialWithAlphaMap, d->alphaMap = s->alphaMap;);
    TPP_COPY_MIXIN(MaterialWithSpecularMap, d->specularMap = s->specularMap;);
    TPP_COPY_MIXIN(MaterialWithEnvMap,
                   d->envMap = s->envMap;
                   d->envMapIntensity = s->envMapIntensity;);
    TPP_COPY_MIXIN(MaterialWithGradientMap, d->gradientMap = s->gradientMap;);
    TPP_COPY_MIXIN(MaterialWithAoMap,
                   d->aoMap = s->aoMap;
                   d->aoMapIntensity = s->aoMapIntensity;);
    TPP_COPY_MIXIN(MaterialWithBumpMap,
                   d->bumpMap = s->bumpMap;
                   d->bumpScale = s->bumpScale;);
    TPP_COPY_MIXIN(MaterialWithLightMap,
                   d->lightMap = s->lightMap;
                   d->lightMapIntensity = s->lightMapIntensity;);
    TPP_COPY_MIXIN(MaterialWithDisplacementMap,
                   d->displacementMap = s->displacementMap;
                   d->displacementScale = s->displacementScale;
                   d->displacementBias = s->displacementBias;);
    TPP_COPY_MIXIN(MaterialWithNormalMap,
                   d->normalMap = s->normalMap;
                   d->normalMapType = s->normalMapType;
                   d->normalScale = s->normalScale;);
    TPP_COPY_MIXIN(MaterialWithMatCap, d->matcap = s->matcap;);
    TPP_COPY_MIXIN(MaterialWithRoughness,
                   d->roughness = s->roughness;
                   d->roughnessMap = s->roughnessMap;);
    TPP_COPY_MIXIN(MaterialWithMetalness,
                   d->metalness = s->metalness;
                   d->metalnessMap = s->metalnessMap;);
    TPP_COPY_MIXIN(MaterialWithThickness,
                   d->thickness = s->thickness;
                   d->thicknessMap = s->thicknessMap;);
    TPP_COPY_MIXIN(MaterialWithClearcoat,
                   d->clearcoat = s->clearcoat;
                   d->clearcoatMap = s->clearcoatMap;
                   d->clearcoatRoughness = s->clearcoatRoughness;
                   d->clearcoatRoughnessMap = s->clearcoatRoughnessMap;
                   d->clearcoatNormalScale = s->clearcoatNormalScale;
                   d->clearcoatNormalMap = s->clearcoatNormalMap;);
    TPP_COPY_MIXIN(MaterialWithTransmission,
                   d->transmission = s->transmission;
                   d->ior = s->ior;
                   d->dispersion = s->dispersion;
                   d->transmissionMap = s->transmissionMap;);
    TPP_COPY_MIXIN(MaterialWithAttenuation,
                   d->attenuationDistance = s->attenuationDistance;
                   d->attenuationColor = s->attenuationColor;);
    TPP_COPY_MIXIN(MaterialWithSheen,
                   d->sheenColor = s->sheenColor;
                   d->sheenRoughness = s->sheenRoughness;
                   d->sheen = s->sheen;);
    TPP_COPY_MIXIN(MaterialWithPbrSpecular,
                   d->specularIntensity = s->specularIntensity;
                   d->specularColor = s->specularColor;);
    TPP_COPY_MIXIN(MaterialWithCombine, d->combine = s->combine;);
    TPP_COPY_MIXIN(MaterialWithDepthPacking, d->depthPacking = s->depthPacking;);
    TPP_COPY_MIXIN(MaterialWithFlatShading, d->flatShading = s->flatShading;);
    TPP_COPY_MIXIN(MaterialWithVertexTangents, d->vertexTangents = s->vertexTangents;);
    TPP_COPY_MIXIN(MaterialWithDefines, d->defines = s->defines;);
    TPP_COPY_MIXIN(MaterialWithMorphTargets,
                   d->morphTargets = s->morphTargets;
                   d->morphNormals = s->morphNormals;);

#undef TPP_COPY_MIXIN
}

Material::~Material() {

    dispose();
}

void Material::setValues(const std::unordered_map<std::string, MaterialValue>& values) {

    if (values.empty()) return;

    std::vector<std::string> unused;

    for (const auto& [key, value] : values) {

        bool used = false;

        if (key == "fog") {

            fog = std::get<bool>(value);
            used = true;

        } else if (key == "blending") {

            blending = std::get<Blending>(value);
            used = true;

        } else if (key == "side") {

            side = std::get<Side>(value);
            used = true;

        } else if (key == "vertexColors") {

            vertexColors = std::get<bool>(value);
            used = true;

        } else if (key == "opacity") {

            opacity = extractFloat(value);
            used = true;

        } else if (key == "transparent") {

            transparent = std::get<bool>(value);
            used = true;

        } else if (key == "blendSrc") {

            blendSrc = std::get<BlendFactor>(value);
            used = true;

        } else if (key == "blendDst") {

            blendDst = std::get<BlendFactor>(value);
            used = true;

        } else if (key == "blendEquation") {

            blendEquation = std::get<BlendEquation>(value);
            used = true;

        } else if (key == "blendSrcAlpha") {

            blendSrcAlpha = std::get<BlendFactor>(value);
            used = true;

        } else if (key == "blendDstAlpha") {

            blendDstAlpha = std::get<BlendFactor>(value);

        } else if (key == "blendEquationAlpha") {

            blendEquationAlpha = std::get<BlendEquation>(value);
            used = true;

        } else if (key == "depthFunc") {

            depthFunc = std::get<DepthFunc>(value);
            used = true;

        } else if (key == "depthTest") {

            depthTest = std::get<bool>(value);
            used = true;

        } else if (key == "depthWrite") {

            depthWrite = std::get<bool>(value);
            used = true;

        } else if (key == "stencilWriteMask") {

            stencilWriteMask = std::get<int>(value);
            used = true;

        } else if (key == "stencilFunc") {

            stencilFunc = std::get<StencilFunc>(value);
            used = true;

        } else if (key == "stencilRef") {

            stencilRef = std::get<int>(value);
            used = true;

        } else if (key == "stencilFuncMask") {

            stencilFuncMask = std::get<int>(value);
            used = true;

        } else if (key == "stencilFail") {

            stencilFail = std::get<StencilOp>(value);
            used = true;

        } else if (key == "stencilZFail") {

            stencilZFail = std::get<StencilOp>(value);
            used = true;

        } else if (key == "stencilZPass") {

            stencilZPass = std::get<StencilOp>(value);
            used = true;

        } else if (key == "stencilWrite") {

            stencilWrite = std::get<int>(value);
            used = true;

        } else if (key == "shadowSide") {

            shadowSide = std::get<Side>(value);
            used = true;

        } else if (key == "colorWrite") {

            colorWrite = std::get<bool>(value);
            used = true;

        } else if (key == "polygonOffset") {

            polygonOffset = std::get<bool>(value);
            used = true;

        } else if (key == "polygonOffsetFactor") {

            polygonOffsetFactor = extractFloat(value);
            used = true;

        } else if (key == "polygonOffsetUnits") {

            polygonOffsetUnits = extractFloat(value);
            used = true;

        } else if (key == "dithering") {

            dithering = std::get<bool>(value);
            used = true;

        } else if (key == "alphaTest") {

            alphaTest = extractFloat(value);
            used = true;

        } else if (key == "alphaToCoverage") {

            alphaToCoverage = std::get<bool>(value);
            used = true;

        } else if (key == "premultipliedAlpha") {

            premultipliedAlpha = std::get<bool>(value);
            used = true;

        } else if (key == "visible") {

            visible = std::get<bool>(value);
            used = true;

        } else if (key == "toneMapped") {

            toneMapped = std::get<bool>(value);
            used = true;

        } else {

            used = setValue(key, value);
        }

        if (!used) {
            unused.emplace_back(key);
        }
    }

    if (!unused.empty()) {
        std::stringstream ss;
        ss << "[";
        for (unsigned i = 0; i < unused.size(); i++) {
            ss << unused[i];
            if (i < unused.size() - 1) {
                ss << ",";
            }
        }
        ss << "]";

        std::cerr << "Unused material values: " << ss.str() << std::endl;
    }
}

bool Material::setValue(const std::string&, const MaterialValue&) {

    return false;
}
