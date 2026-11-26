// How the cathedral is shown on the GL renderer: pale pine planks in the
// hall's low sun (Hall.hpp), and the room's own light to fill the shadows.
//
// The wood is made in the shader from each plank's own coordinates: every
// board is cut from a log of its own, with its pith somewhere beside it, so its
// long faces show the growth rings as fine stripes and shallow arches, and its
// ends show the rings themselves. The end grain is a little darker and rougher.
// Tint, ring spacing and where the log was are drawn per plank. All of it is
// meant to be felt rather than seen: a detail one can name at a glance is too
// strong.

#ifndef PLANKCATHEDRAL_LOOK_HPP
#define PLANKCATHEDRAL_LOOK_HPP

#include "threepp/threepp.hpp"

#include "Cathedral.hpp"
#include "Hall.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

namespace planks {

    // `sun` is the hall's sun (sunGlsl): it reaches a plank only through the windows
    inline std::shared_ptr<threepp::MeshStandardMaterial> woodMaterial(const std::string& sun) {
        using namespace threepp;

        auto m = MeshStandardMaterial::create();
        m->color = Color(1.f, 1.f, 1.f);// the plank's tint comes per instance
        m->roughness = 0.62f;
        m->metalness = 0.f;
        m->customProgramCacheKey = "plank-cathedral-wood";
        m->onBeforeCompile = [sun](Shader& s) {
            addSun(s, sun);
            const auto replace = [](std::string& src, const std::string& what, const std::string& with) {
                const auto at = src.find(what);
                if (at != std::string::npos) src.replace(at, what.size(), with);
            };
            replace(s.vertexShader, "#include <common>", R"(#include <common>
varying vec3 vWoodPos;
varying vec3 vWoodNormal;
varying float vWoodSeed;)");
            replace(s.vertexShader, "#include <begin_vertex>", R"(#include <begin_vertex>
vWoodPos = position;
vWoodNormal = normal;
vWoodSeed = float(gl_InstanceID);)");

            replace(s.fragmentShader, "#include <common>", R"(#include <common>
varying vec3 vWoodPos;
varying vec3 vWoodNormal;
varying float vWoodSeed;

float woodHash(float n) { return fract(sin(n * 78.233 + 0.5) * 43758.5453); }
float woodHash3(vec3 p) {
    p = fract(p * 0.3183099 + vec3(0.71, 0.113, 0.419));
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float woodNoise(vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(woodHash3(i), woodHash3(i + vec3(1, 0, 0)), f.x),
                   mix(woodHash3(i + vec3(0, 1, 0)), woodHash3(i + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(woodHash3(i + vec3(0, 0, 1)), woodHash3(i + vec3(1, 0, 1)), f.x),
                   mix(woodHash3(i + vec3(0, 1, 1)), woodHash3(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
// 1 on the plank's two small end faces, where the fibres are cut
float woodEnd() {
    return smoothstep(0.55, 0.85, abs(normalize(vWoodNormal).x));
})");

            replace(s.fragmentShader, "#include <color_fragment>", R"(#include <color_fragment>
{
    // the board's own log: its pith off to one side of the wide face, the board
    // cut a little across the grain so that the rings rise into arches
    float h0 = woodHash(vWoodSeed);
    float h1 = woodHash(vWoodSeed + 17.31);
    float h2 = woodHash(vWoodSeed + 41.07);
    float h3 = woodHash(vWoodSeed + 73.93);
    vec3 q = vWoodPos;
    vec2 pith = vec2(mix(0.035, 0.11, h0) * (h1 < 0.5 ? -1.0 : 1.0), mix(-0.02, 0.02, h2));
    float r = length(q.yz - pith);
    r += (h3 - 0.5) * 0.08 * q.x;
    r += 0.0007 * (woodNoise(vec3(q.x * 22.0, q.y * 70.0, q.z * 70.0) + h0 * 37.0) - 0.5);
    float perMetre = mix(260.0, 430.0, h2);
    float ring = fract(r * perMetre);
    float late = smoothstep(0.58, 0.84, ring) * (1.0 - smoothstep(0.9, 1.0, ring));
    vec3 grain = mix(vec3(1.0), vec3(0.90, 0.87, 0.82), late);
    // fine fibres running the length of the board
    grain *= 1.0 - 0.035 * woodNoise(vec3(q.x * 30.0, q.y * 1400.0, q.z * 1400.0) + h1 * 19.0);
    // cut fibres drink light: the ends are darker
    grain *= mix(1.0, 0.8, woodEnd());
    diffuseColor.rgb *= grain;
})");

            replace(s.fragmentShader, "#include <roughnessmap_fragment>", R"(#include <roughnessmap_fragment>
roughnessFactor = min(1.0, roughnessFactor + 0.18 * woodEnd());)");
        };
        return m;
    }

    // Pale pine, a little different on every board.
    inline threepp::Color plankTint(Rng& rng) {
        using namespace threepp;
        const float value = 1.f + rng.uniform(-0.06f, 0.06f);
        const float warm = rng.uniform(-1.f, 1.f);
        const float r = 0.88f * value * (1.f + 0.012f * warm);
        const float g = 0.79f * value * (1.f - 0.004f * warm);
        const float b = 0.62f * value * (1.f - 0.035f * warm);
        Color c;
        c.setRGB(r, g, b, ColorSpace::sRGB);
        return c;
    }

    // Fits an orthographic shadow camera around a box, seen from the light.
    inline void fitShadow(threepp::DirectionalLight& light, threepp::Object3D& target, const threepp::Box3& box) {
        using namespace threepp;
        Vector3 centre;
        box.getCenter(centre);
        target.position.copy(centre);
        target.updateMatrixWorld();

        const Vector3 eye = light.position;
        Vector3 f = Vector3(centre).sub(eye).normalize();
        Vector3 r = Vector3().crossVectors(f, Vector3(0, 1, 0)).normalize();
        Vector3 u = Vector3().crossVectors(r, f);
        float minX = 1e9f, maxX = -1e9f, minY = 1e9f, maxY = -1e9f, minZ = 1e9f, maxZ = -1e9f;
        for (int i = 0; i < 8; ++i) {
            const Vector3 c((i & 1) ? box.max().x : box.min().x, (i & 2) ? box.max().y : box.min().y, (i & 4) ? box.max().z : box.min().z);
            const Vector3 d = Vector3(c).sub(eye);
            minX = std::min(minX, d.dot(r)), maxX = std::max(maxX, d.dot(r));
            minY = std::min(minY, d.dot(u)), maxY = std::max(maxY, d.dot(u));
            minZ = std::min(minZ, d.dot(f)), maxZ = std::max(maxZ, d.dot(f));
        }
        auto* cam = light.shadow->camera->as<OrthographicCamera>();
        const float pad = 0.03f;
        cam->left = minX - pad;
        cam->right = maxX + pad;
        cam->bottom = minY - pad;
        cam->top = maxY + pad;
        cam->nearPlane = std::max(0.01f, minZ - 0.5f);
        cam->farPlane = maxZ + 0.5f;
        cam->updateProjectionMatrix();
    }

}// namespace planks

#endif//PLANKCATHEDRAL_LOOK_HPP
