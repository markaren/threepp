// Material parameter extraction and uniform packing for the Dawn renderer.

#include "DawnMaterials.hpp"

#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/MeshLambertMaterial.hpp"
#include "threepp/materials/MeshPhongMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/materials/MeshToonMaterial.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/materials/PointsMaterial.hpp"
#include "threepp/materials/SpriteMaterial.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/materials/ShadowMaterial.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/core/BufferGeometry.hpp"

#include <cstring>

using namespace threepp;
using namespace threepp::dawn;

namespace threepp::dawn {

MaterialParams extractMaterialParams(Material* rawMat, BufferGeometry* geometry) {
    MaterialParams p;
    p.opacity = rawMat->opacity;

    if (auto m = dynamic_cast<MeshStandardMaterial*>(rawMat)) {
        p.features |= ShaderFeatures::Lighting | ShaderFeatures::PBR;
        p.diffuse = m->color;
        p.roughness = m->roughness;
        p.metalness = m->metalness;
        p.emissive = m->emissive;
        if (m->map) { p.diffuseMap = m->map.get(); p.features |= ShaderFeatures::Texture; }
        if (m->normalMap) {
            p.normalMap = m->normalMap.get();
            p.normalScale = m->normalScale;
            p.features |= ShaderFeatures::NormalMap;
        }
        if (m->emissiveMap) { p.emissiveMap = m->emissiveMap.get(); p.features |= ShaderFeatures::EmissiveMap; }
        if (m->roughnessMap) { p.roughnessMap = m->roughnessMap.get(); p.features |= ShaderFeatures::RoughnessMap; }
        if (m->metalnessMap) { p.metalnessMap = m->metalnessMap.get(); p.features |= ShaderFeatures::MetalnessMap; }
        if (m->aoMap) { p.aoMap = m->aoMap.get(); p.aoMapIntensity = m->aoMapIntensity; p.features |= ShaderFeatures::AOMap; }
        if (m->alphaMap) { p.alphaMap = m->alphaMap.get(); p.features |= ShaderFeatures::AlphaMap; }
        if (m->lightMap) { p.lightMap = m->lightMap.get(); p.features |= ShaderFeatures::LightMap; }
        if (m->bumpMap) { p.bumpMap = m->bumpMap.get(); p.bumpScale = m->bumpScale; p.features |= ShaderFeatures::BumpMap; }
        if (m->displacementMap) { p.displacementMap = m->displacementMap.get(); p.displacementScale = m->displacementScale; p.features |= ShaderFeatures::DisplacementMap; }
        if (m->envMap) { p.envMap = m->envMap.get(); p.envMapIntensity = m->envMapIntensity; p.features |= ShaderFeatures::EnvMap; }
    } else if (auto m = dynamic_cast<MeshPhongMaterial*>(rawMat)) {
        p.features |= ShaderFeatures::Lighting | ShaderFeatures::Specular;
        p.diffuse = m->color;
        p.specularColor = m->specular;
        p.shininess = m->shininess;
        p.emissive = m->emissive;
        if (m->map) { p.diffuseMap = m->map.get(); p.features |= ShaderFeatures::Texture; }
        if (m->normalMap) { p.normalMap = m->normalMap.get(); p.normalScale = m->normalScale; p.features |= ShaderFeatures::NormalMap; }
        if (m->emissiveMap) { p.emissiveMap = m->emissiveMap.get(); p.features |= ShaderFeatures::EmissiveMap; }
        if (m->aoMap) { p.aoMap = m->aoMap.get(); p.aoMapIntensity = m->aoMapIntensity; p.features |= ShaderFeatures::AOMap; }
        if (m->alphaMap) { p.alphaMap = m->alphaMap.get(); p.features |= ShaderFeatures::AlphaMap; }
        if (m->specularMap) { p.specularMap = m->specularMap.get(); p.features |= ShaderFeatures::SpecularMap; }
        if (m->lightMap) { p.lightMap = m->lightMap.get(); p.features |= ShaderFeatures::LightMap; }
        if (m->bumpMap) { p.bumpMap = m->bumpMap.get(); p.bumpScale = m->bumpScale; p.features |= ShaderFeatures::BumpMap; }
    } else if (auto m = dynamic_cast<MeshToonMaterial*>(rawMat)) {
        p.features |= ShaderFeatures::Lighting;
        p.diffuse = m->color;
        p.emissive = m->emissive;
        if (m->map) { p.diffuseMap = m->map.get(); p.features |= ShaderFeatures::Texture; }
        if (m->emissiveMap) { p.emissiveMap = m->emissiveMap.get(); p.features |= ShaderFeatures::EmissiveMap; }
        if (m->normalMap) { p.normalMap = m->normalMap.get(); p.normalScale = m->normalScale; p.features |= ShaderFeatures::NormalMap; }
        if (m->bumpMap) { p.bumpMap = m->bumpMap.get(); p.bumpScale = m->bumpScale; p.features |= ShaderFeatures::BumpMap; }
        if (m->gradientMap) { p.gradientMap = m->gradientMap.get(); p.features |= ShaderFeatures::GradientMap; }
    } else if (auto m = dynamic_cast<MeshLambertMaterial*>(rawMat)) {
        p.features |= ShaderFeatures::Lighting;
        p.diffuse = m->color;
        p.emissive = m->emissive;
        if (m->map) { p.diffuseMap = m->map.get(); p.features |= ShaderFeatures::Texture; }
    } else if (auto m = dynamic_cast<MeshBasicMaterial*>(rawMat)) {
        p.diffuse = m->color;
        if (m->map) { p.diffuseMap = m->map.get(); p.features |= ShaderFeatures::Texture; }
    } else if (auto m = dynamic_cast<LineBasicMaterial*>(rawMat)) {
        p.diffuse = m->color;
    } else if (auto m = dynamic_cast<PointsMaterial*>(rawMat)) {
        p.diffuse = m->color;
        if (m->map) { p.diffuseMap = m->map.get(); p.features |= ShaderFeatures::Texture; }
    } else if (auto m = dynamic_cast<SpriteMaterial*>(rawMat)) {
        p.diffuse = m->color;
        p.opacity = m->opacity;
        if (m->map) { p.diffuseMap = m->map.get(); p.features |= ShaderFeatures::Texture; }
    } else if (dynamic_cast<ShadowMaterial*>(rawMat)) {
        p.skip = true;
        return p;
    } else if (auto m = dynamic_cast<ShaderMaterial*>(rawMat)) {
        // Detect custom WGSL shaders (no GLSL markers)
        bool isGLSL = m->vertexShader.find("gl_Position") != std::string::npos ||
                      m->fragmentShader.find("gl_FragColor") != std::string::npos;
        if (!isGLSL) {
            p.isCustomShader = true;
        }

        // Try to extract diffuse color from uniforms
        if (m->uniforms.count("uColor") && m->uniforms.at("uColor").hasValue()) {
            try {
                auto& val = const_cast<Uniform&>(m->uniforms.at("uColor")).value();
                if (auto* c = std::get_if<Color>(&val)) {
                    p.diffuse = *c;
                }
            } catch (...) {}
        } else if (m->uniforms.count("color") && m->uniforms.at("color").hasValue()) {
            try {
                auto& val = const_cast<Uniform&>(m->uniforms.at("color")).value();
                if (auto* c = std::get_if<Color>(&val)) {
                    p.diffuse = *c;
                }
            } catch (...) {}
        }
    } else if (auto cm = dynamic_cast<MaterialWithColor*>(rawMat)) {
        p.diffuse = cm->color;
    }

    // Vertex colors
    if (rawMat->vertexColors && geometry->hasAttribute("color")) {
        p.features |= ShaderFeatures::VertexColors;
    }

    return p;
}

void packMaterialUniforms(float* data, const MaterialParams& params,
                          const FrameContext& ctx, Material* rawMat) {
    std::memset(data, 0, MATERIAL_UNIFORM_SIZE);

    // diffuse (vec4, offset 0)
    data[0] = params.diffuse.r;
    data[1] = params.diffuse.g;
    data[2] = params.diffuse.b;
    data[3] = 1.0f;

    // specular (vec4, offset 4): rgb + shininess
    data[4] = params.specularColor.r;
    data[5] = params.specularColor.g;
    data[6] = params.specularColor.b;
    data[7] = params.shininess;

    // roughnessMetalnessOpacity (vec4, offset 8): roughness, metalness, opacity, displacementScale
    data[8] = params.roughness;
    data[9] = params.metalness;
    data[10] = params.opacity;
    data[11] = params.displacementScale;

    // emissive (vec4, offset 12): rgb + envMapIntensity
    data[12] = params.emissive.r;
    data[13] = params.emissive.g;
    data[14] = params.emissive.b;
    data[15] = params.envMapIntensity;

    // flags (vec4, offset 16): aoMapIntensity, bumpScale, normalScale.xy
    data[16] = params.aoMapIntensity;
    data[17] = params.bumpScale;
    data[18] = params.normalScale.x;
    data[19] = params.normalScale.y;

    // fogColor (vec4, offset 20)
    data[20] = ctx.fogColor.r;
    data[21] = ctx.fogColor.g;
    data[22] = ctx.fogColor.b;
    data[23] = 1.0f;

    // fogParams (vec4, offset 24): near, far, density, toneMappingExposure
    data[24] = ctx.fogNear;
    data[25] = ctx.fogFar;
    data[26] = ctx.fogDensity;
    data[27] = ctx.toneMappingExposure;

    // clipPlane (vec4, offset 28): normal.xyz + constant
    if (ctx.localClippingEnabled && !rawMat->clippingPlanes.empty()) {
        auto& cp = rawMat->clippingPlanes[0];
        data[28] = cp.normal.x;
        data[29] = cp.normal.y;
        data[30] = cp.normal.z;
        data[31] = cp.constant;
    }
}

}// namespace threepp::dawn
