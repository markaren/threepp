
#include "threepp/materials/Material.hpp"

#include "threepp/math/MathUtils.hpp"

#include <iostream>
#include <sstream>

using namespace threepp;

Material::Material()
    : uuid_(math::generateUUID()) {}

std::string Material::uuid() const {

    return uuid_;
}

void Material::dispose() {
    if (!disposed_) {
        disposed_ = true;
        dispatchEvent("dispose", this);
    }
}

void Material::needsUpdate() {

    this->version++;
}

void Material::copyInto(Material* m) const {

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

    const auto& srcPlanes = clippingPlanes;
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

            opacity = std::get<float>(value);
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

            polygonOffsetFactor = std::get<float>(value);
            used = true;

        } else if (key == "polygonOffsetUnits") {

            polygonOffsetUnits = std::get<float>(value);
            used = true;

        } else if (key == "dithering") {

            dithering = std::get<bool>(value);
            used = true;

        } else if (key == "alphaTest") {

            alphaTest = std::get<float>(value);
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

bool Material::setValue(const std::string& key, const MaterialValue& value) {

    return false;
}
