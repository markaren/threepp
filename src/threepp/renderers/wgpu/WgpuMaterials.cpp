// Material parameter extraction and uniform packing for the Wgpu renderer.

#include "WgpuMaterials.hpp"

#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/MeshDepthMaterial.hpp"
#include "threepp/materials/MeshLambertMaterial.hpp"
#include "threepp/materials/MeshNormalMaterial.hpp"
#include "threepp/materials/MeshPhongMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/materials/MeshToonMaterial.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/materials/LineDashedMaterial.hpp"
#include "threepp/materials/PointsMaterial.hpp"
#include "threepp/materials/SpriteMaterial.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/materials/ShadowMaterial.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/textures/Texture.hpp"

#include <cstring>

using namespace threepp;
using namespace threepp::wgpu;

namespace threepp::wgpu {

MaterialParams extractMaterialParams(Material* rawMat, BufferGeometry* geometry, Texture* sceneEnvFallback) {
    MaterialParams p;
    p.opacity = rawMat->opacity;

    if (auto m = dynamic_cast<MeshStandardMaterial*>(rawMat)) {
        p.features |= ShaderFeatures::Lighting | ShaderFeatures::PBR;
        p.diffuse = m->color;
        p.roughness = m->roughness;
        p.metalness = m->metalness;
        p.emissive = Color(m->emissive.r * m->emissiveIntensity,
                           m->emissive.g * m->emissiveIntensity,
                           m->emissive.b * m->emissiveIntensity);
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
        if (m->envMap) {
            p.envMap = m->envMap.get();
            p.envMapIntensity = m->envMapIntensity;
            const bool isCube = m->envMap->mapping == Mapping::CubeReflection || m->envMap->mapping == Mapping::CubeRefraction;
            p.features |= isCube ? ShaderFeatures::EnvMapCube : ShaderFeatures::EnvMap;
        } else if (sceneEnvFallback) {
            p.envMap = sceneEnvFallback;
            p.envMapIntensity = 1.0f;
            const bool isCube = sceneEnvFallback->mapping == Mapping::CubeReflection || sceneEnvFallback->mapping == Mapping::CubeRefraction;
            p.features |= isCube ? ShaderFeatures::EnvMapCube : ShaderFeatures::EnvMap;
        }
        if (auto t = dynamic_cast<MeshPhysicalMaterial*>(rawMat)) {
            p.transmission = t->transmission;
            p.ior = t->ior;
            p.thickness = t->thickness;
            p.attenuationDistance = t->attenuationDistance;
            p.attenuationColor = t->attenuationColor;
            if (p.transmission > 0.0f) p.features |= ShaderFeatures::Transmission;

            // KHR_materials_specular: dielectric F0 = 0.04 * specularColor * specularIntensity.
            // Reuses the specularAndShininess uniform slot (rgb = color, w = intensity).
            p.specularColor = t->specularColor;
            p.specularIntensity = t->specularIntensity;

            p.sheenColor = t->sheen.value_or(t->sheenColor);
            p.sheenRoughness = t->sheenRoughness;
            if (p.sheenColor.r > 0.0f || p.sheenColor.g > 0.0f || p.sheenColor.b > 0.0f) {
                p.features |= ShaderFeatures::Sheen;
            }
        }
    } else if (auto m = dynamic_cast<MeshPhongMaterial*>(rawMat)) {
        p.features |= ShaderFeatures::Lighting | ShaderFeatures::Specular;
        p.diffuse = m->color;
        p.specularColor = m->specular;
        p.shininess = m->shininess;
        p.emissive = Color(m->emissive.r * m->emissiveIntensity,
                           m->emissive.g * m->emissiveIntensity,
                           m->emissive.b * m->emissiveIntensity);
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
        p.emissive = Color(m->emissive.r * m->emissiveIntensity,
                           m->emissive.g * m->emissiveIntensity,
                           m->emissive.b * m->emissiveIntensity);
        if (m->map) { p.diffuseMap = m->map.get(); p.features |= ShaderFeatures::Texture; }
        if (m->emissiveMap) { p.emissiveMap = m->emissiveMap.get(); p.features |= ShaderFeatures::EmissiveMap; }
        if (m->normalMap) { p.normalMap = m->normalMap.get(); p.normalScale = m->normalScale; p.features |= ShaderFeatures::NormalMap; }
        if (m->bumpMap) { p.bumpMap = m->bumpMap.get(); p.bumpScale = m->bumpScale; p.features |= ShaderFeatures::BumpMap; }
        if (m->gradientMap) { p.gradientMap = m->gradientMap.get(); p.features |= ShaderFeatures::GradientMap; }
    } else if (auto m = dynamic_cast<MeshLambertMaterial*>(rawMat)) {
        p.features |= ShaderFeatures::Lighting;
        p.diffuse = m->color;
        p.emissive = Color(m->emissive.r * m->emissiveIntensity,
                           m->emissive.g * m->emissiveIntensity,
                           m->emissive.b * m->emissiveIntensity);
        if (m->map) { p.diffuseMap = m->map.get(); p.features |= ShaderFeatures::Texture; }
        if (m->envMap) {
            p.envMap = m->envMap.get();
            p.envMapIntensity = m->envMapIntensity;
            const bool isCube = m->envMap->mapping == Mapping::CubeReflection || m->envMap->mapping == Mapping::CubeRefraction;
            p.features |= isCube ? ShaderFeatures::EnvMapCube : ShaderFeatures::EnvMap;
        }
    } else if (dynamic_cast<MeshNormalMaterial*>(rawMat)) {
        p.features |= ShaderFeatures::NormalVis;
    } else if (dynamic_cast<MeshDepthMaterial*>(rawMat)) {
        p.features |= ShaderFeatures::DepthVis;
    } else if (auto m = dynamic_cast<MeshBasicMaterial*>(rawMat)) {
        p.diffuse = m->color;
        if (m->map) { p.diffuseMap = m->map.get(); p.features |= ShaderFeatures::Texture; }
    } else if (auto m = dynamic_cast<LineDashedMaterial*>(rawMat)) {
        p.features |= ShaderFeatures::LineDashed;
        p.diffuse = m->color;
        // Pack dash parameters into specularColor/shininess fields (unused for lines)
        p.specularColor = Color(m->dashSize, m->dashSize + m->gapSize, m->scale);
    } else if (auto m = dynamic_cast<LineBasicMaterial*>(rawMat)) {
        p.diffuse = m->color;
    } else if (auto m = dynamic_cast<PointsMaterial*>(rawMat)) {
        p.diffuse = m->color;
        if (m->map) { p.diffuseMap = m->map.get(); p.features |= ShaderFeatures::Texture; }
    } else if (auto m = dynamic_cast<SpriteMaterial*>(rawMat)) {
        p.diffuse = m->color;
        p.opacity = m->opacity;
        if (m->map) { p.diffuseMap = m->map.get(); p.features |= ShaderFeatures::Texture; }
    } else if (auto m = dynamic_cast<ShadowMaterial*>(rawMat)) {
        // ShadowMaterial: renders a transparent surface showing only shadow darkening.
        // Needs lighting + shadow, but fragment outputs shadow attenuation only.
        p.features |= ShaderFeatures::ShadowMat | ShaderFeatures::Lighting;
        p.diffuse = m->color;
    } else if (auto m = dynamic_cast<ShaderMaterial*>(rawMat)) {
        // Detect GLSL (three.js) vs native WGSL shaders
        bool isGLSL = m->vertexShader.find("gl_Position") != std::string::npos ||
                      m->fragmentShader.find("gl_FragColor") != std::string::npos;
#ifdef THREEPP_WGPU_GLSL_COMPAT
        // With GLSL compat enabled, route GLSL shaders through the translation pipeline
        p.isCustomShader = true;
#else
        if (!isGLSL) {
            p.isCustomShader = true;
        }
#endif
        (void)isGLSL;

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

    // specular (vec4, offset 4): rgb + (PBR: specularIntensity, else: shininess)
    data[4] = params.specularColor.r;
    data[5] = params.specularColor.g;
    data[6] = params.specularColor.b;
    data[7] = (params.features & ShaderFeatures::PBR) ? params.specularIntensity : params.shininess;

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

    // transmissionParams (vec4, offset 32): transmission, ior, thickness, samplerWidth
    data[32] = params.transmission;
    data[33] = params.ior;
    data[34] = params.thickness;
    data[35] = ctx.transmissionTexW;

    // attenuationParams (vec4, offset 36): color.rgb, distance
    data[36] = params.attenuationColor.r;
    data[37] = params.attenuationColor.g;
    data[38] = params.attenuationColor.b;
    data[39] = params.attenuationDistance;

    // sheenColorAndRoughness (vec4, offset 40): sheenColor.rgb, sheenRoughness
    data[40] = params.sheenColor.r;
    data[41] = params.sheenColor.g;
    data[42] = params.sheenColor.b;
    data[43] = params.sheenRoughness;

    // Per-map UV transforms (vec4×2 each, offsets 44-131): KHR_texture_transform.
    // Order matches the WGSL MaterialUniforms struct: map, normalMap, roughnessMap,
    // metalnessMap, emissiveMap, aoMap, alphaMap, lightMap, specularMap, bumpMap,
    // displacementMap. Each transform is row0/row1 of the 3x3 affine.
    auto writeUvT = [&](size_t floatOff, Texture* tex) {
        if (tex) {
            if (tex->matrixAutoUpdate) tex->updateMatrix();
            const auto& te = tex->matrix.elements;
            // T1.w (floatOff+7) carries the texCoord channel index (0 = uv, 1 = uv2)
            // for KHR_texture_transform texCoord override. Shader compares > 0.5.
            float ch = (tex->texCoord >= 1) ? 1.0f : 0.0f;
            data[floatOff + 0] = te[0]; data[floatOff + 1] = te[3]; data[floatOff + 2] = te[6]; data[floatOff + 3] = 0.0f;
            data[floatOff + 4] = te[1]; data[floatOff + 5] = te[4]; data[floatOff + 6] = te[7]; data[floatOff + 7] = ch;
        } else {
            data[floatOff + 0] = 1.0f; data[floatOff + 1] = 0.0f; data[floatOff + 2] = 0.0f; data[floatOff + 3] = 0.0f;
            data[floatOff + 4] = 0.0f; data[floatOff + 5] = 1.0f; data[floatOff + 6] = 0.0f; data[floatOff + 7] = 0.0f;
        }
    };
    writeUvT(44,  params.diffuseMap);
    writeUvT(52,  params.normalMap);
    writeUvT(60,  params.roughnessMap);
    writeUvT(68,  params.metalnessMap);
    writeUvT(76,  params.emissiveMap);
    writeUvT(84,  params.aoMap);
    writeUvT(92,  params.alphaMap);
    writeUvT(100, params.lightMap);
    writeUvT(108, params.specularMap);
    writeUvT(116, params.bumpMap);
    writeUvT(124, params.displacementMap);
}

}// namespace threepp::wgpu
