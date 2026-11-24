// The valley's air, for the GL renderer: every lit material's fog step is replaced (onBeforeCompile) by air that is
// denser low in the valley and thins with height (exponential height fog, integrated exactly along the view ray),
// and lit: the sky's light from all round plus the sun's, scattered forward (Henyey-Greenstein), so the air glows
// toward the sun and the far woods sink into a blue-green haze. VulkanRenderer and its path tracer have their own
// air medium (scene.fog with setHeightFog), which also casts the canopy's shadows as shafts.

#ifndef THREEPP_EXAMPLE_WATERMILL_ATMOSPHERE_HPP
#define THREEPP_EXAMPLE_WATERMILL_ATMOSPHERE_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Mesh.hpp"

#include <functional>
#include <string>
#include <unordered_set>

namespace mill {

    struct AirLook {
        threepp::Vector3 sunDirection{0, 1, 0};// unit, toward the sun
        threepp::Vector3 sun{1, 1, 1};         // the sun's light the air scatters (radiance scale)
        threepp::Vector3 ambient{0.4f, 0.45f, 0.5f};// the sky's light the air scatters, from all round
        float density = 0.012f;                // extinction per metre at the base height
        float base = 0.f;                      // y where the density is `density`
        float falloff = 25.f;                  // m for the density to fall by e
        float g = 0.6f;                        // forward scattering
    };

    inline const char* airFogChunk() {
        return R"(#ifdef USE_FOG
{
    vec3 airRay = (vec4(-vViewPosition, 0.0) * viewMatrix).xyz;
    float airDist = length(airRay);
    vec3 airDir = airRay / max(airDist, 1e-4);
    float airK = airDir.y / uAirFalloff;
    float airAtCam = uAirDensity * exp(-(cameraPosition.y - uAirBase) / uAirFalloff);
    float airOd = abs(airK * airDist) < 1e-3 ? airAtCam * airDist : airAtCam * (1.0 - exp(-airK * airDist)) / airK;
    float airT = exp(-airOd);
    float airMu = dot(airDir, uAirSunDir);
    float airG2 = uAirG * uAirG;
    float airHg = (1.0 - airG2) / (12.566 * pow(max(1.0 + airG2 - 2.0 * uAirG * airMu, 1e-4), 1.5));
    vec3 airIn = uAirAmbient + uAirSun * airHg;
    gl_FragColor.rgb = gl_FragColor.rgb * airT + airIn * (1.0 - airT);
}
#endif)";
    }

    // give every lit material under `root` the valley's air (keeping any hook it already has)
    inline void applyAir(threepp::Object3D& root, const AirLook& air) {
        using namespace threepp;
        std::unordered_set<Material*> done;
        root.traverse([&](Object3D& o) {
            auto* mesh = o.as<Mesh>();
            if (!mesh) return;
            for (const auto& material : mesh->materials()) {
                auto* m = dynamic_cast<MeshStandardMaterial*>(material.get());
                if (!m || done.count(m) || !m->fog) continue;
                done.insert(m);
                auto previous = m->onBeforeCompile;
                m->onBeforeCompile = [previous](Shader& s) {
                    if (previous) previous(s);
                    const std::string from = "#include <fog_fragment>";
                    const auto at = s.fragmentShader.find(from);
                    if (at != std::string::npos) s.fragmentShader.replace(at, from.size(), airFogChunk());
                    const std::string pars = "#include <fog_pars_fragment>";
                    const auto pat = s.fragmentShader.find(pars);
                    if (pat != std::string::npos)
                        s.fragmentShader.replace(pat, pars.size(), std::string(pars) + R"(
uniform vec3 uAirSunDir;
uniform vec3 uAirSun;
uniform vec3 uAirAmbient;
uniform float uAirDensity;
uniform float uAirBase;
uniform float uAirFalloff;
uniform float uAirG;)");
                };
                if (previous && m->customProgramCacheKey.empty()) m->customProgramCacheKey = "hook" + std::to_string(m->id);
                m->customProgramCacheKey += "+air";
                m->customUniforms["uAirSunDir"] = Uniform(air.sunDirection);
                m->customUniforms["uAirSun"] = Uniform(air.sun);
                m->customUniforms["uAirAmbient"] = Uniform(air.ambient);
                m->customUniforms["uAirDensity"] = Uniform(air.density);
                m->customUniforms["uAirBase"] = Uniform(air.base);
                m->customUniforms["uAirFalloff"] = Uniform(air.falloff);
                m->customUniforms["uAirG"] = Uniform(air.g);
                m->needsUpdate();
            }
        });
    }

}// namespace mill

#endif
