// The hall the cathedral stands in: a large, old exhibition hall in the late
// afternoon. A parquet floor at y = 0, warm off-white plaster walls with flat
// pilasters, a dark timber ceiling, and high along the north wall a row of tall
// arched windows with simple frames, through which a low sun comes in. The
// windows sit high, as in a gallery, so that from the cameras' low eyes the
// cathedral stands against the plain wall below them, not against the light.
//
// The sun is the scene's one shadowed key light. Where it lands is worked out
// in the shaders rather than in a shadow map: a point is in the sun when the
// ray from it toward the sun passes through a window, both faces of the wall's
// depth and between the frame's bars, its edge as soft as the sun's disc at
// that distance. So the window patches on the floor are exact at any size,
// and the shadow map stays fitted tightly to the planks.
//
// Everything is made here in code; nothing is loaded from disk.

#ifndef PLANKCATHEDRAL_HALL_HPP
#define PLANKCATHEDRAL_HALL_HPP

#include "threepp/threepp.hpp"

#include "threepp/geometries/ExtrudeGeometry.hpp"
#include "threepp/renderers/gl/PMREMGenerator.hpp"
#include "threepp/renderers/shaders/ShaderChunk.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace planks {

    struct Hall {
        float x0 = -15, x1 = 15;// the west and east walls' inner faces
        float zN = -7.5f, zS = 8;// the north (window) and south walls' inner faces
        float height = 11.5f;   // to the ceiling
        float wall = 0.4f;      // the walls' thickness: the windows' reveals
        // the windows in the north wall
        int windows = 6;
        float firstX = 0;   // the westmost window's centre
        float bay = 5;      // from one window's centre to the next
        float width = 3.4f; // of an opening
        float sill = 4.0f;
        float spring = 8.3f;// where the round arch starts; its top is half a width higher
        // the frames: two mullions, three transoms
        float bar = 0.035f;// half a bar's width
        // the sun, unit, from the floor toward it
        threepp::Vector3 toSun{-0.68f, 0.41f, -0.61f};

        [[nodiscard]] float windowX(int k) const { return firstX + bay * static_cast<float>(k); }
        [[nodiscard]] float mullion() const { return width / 6; }// its distance from the opening's centre
        [[nodiscard]] float transom(int k) const { return (spring - sill) * static_cast<float>(k + 1) / 3; }// above the sill
    };

    // The hall around a building whose centre is `centre`: the sun from `azimuthDeg` west of north, `elevationDeg`
    // high, and the windows placed so that the ray from `anchor` toward the sun passes through the middle of one,
    // so the building stands in a beam.
    inline Hall makeHall(const threepp::Vector3& centre, const threepp::Vector3& anchor, float azimuthDeg, float elevationDeg) {
        using namespace threepp;
        Hall h;
        const float a = azimuthDeg * math::DEG2RAD, e = elevationDeg * math::DEG2RAD;
        h.toSun.set(-std::sin(a) * std::cos(e), std::sin(e), -std::cos(a) * std::cos(e));
        h.zN = centre.z - 7.5f;
        h.zS = h.zN + 15.5f;
        // where the anchor's ray meets the wall's inner face, shifted by half of what the reveal takes off the far side
        const float s = (h.zN - anchor.z) / h.toSun.z;
        const float onWall = anchor.x + s * h.toSun.x;
        const float reveal = h.wall * std::abs(h.toSun.x / h.toSun.z);
        const float lit = onWall - 0.5f * reveal * (h.toSun.x < 0 ? 1.f : -1.f);
        // the row of windows centred on the building, one of them on `lit`
        const float k = std::round((centre.x - lit) / h.bay - 0.5f * static_cast<float>(h.windows - 1));
        h.firstX = lit + k * h.bay;
        const float end = 0.5f * h.width + 1.5f;
        h.x0 = h.firstX - end;
        h.x1 = h.windowX(h.windows - 1) + end;
        return h;
    }

    // GLSL: hallSun(p), how much of the sun reaches the world point p (0 to 1).
    inline std::string sunGlsl(const Hall& h) {
        char b[4096];
        std::snprintf(b, sizeof(b), R"(
const vec3 HALL_SUN = vec3(%.6f, %.6f, %.6f);
// the signed distance from a point on the window wall to the nearest opening (negative inside it)
float hallOpening(vec2 q) {
    float k = clamp(floor((q.x - %.6f) / %.6f + 0.5), 0.0, %.1f);
    float u = q.x - (%.6f + k * %.6f);
    float v = q.y - %.6f;
    float rect = max(abs(u) - %.6f, max(-v, v - %.6f));
    float arch = length(vec2(u, v - %.6f)) - %.6f;// a whole disc: its lower half is inside the rectangle
    return min(rect, arch);
}
// how much of the sun's disc (width w) the frame's bars leave at a point of the wall's middle
float hallBars(vec2 q, float w) {
    float k = clamp(floor((q.x - %.6f) / %.6f + 0.5), 0.0, %.1f);
    float u = q.x - (%.6f + k * %.6f);
    float v = q.y - %.6f;
    float d = abs(abs(u) - %.6f);
    d = min(d, abs(v - %.6f));
    d = min(d, abs(v - %.6f));
    d = min(d, abs(v - %.6f));
    // a bar of width 2b across a sun of width w: the part of the disc it covers
    float b = %.6f;
    float covered = clamp((min(d + b, 0.5 * w) - max(d - b, -0.5 * w)) / w, 0.0, 1.0);
    return 1.0 - covered;
}
float hallSun(vec3 p) {
    // the sun's disc, half a degree: an edge this far from the wall is this soft
    const float disc = 0.0093;
    float m = 1.0;
    float s0 = (%.6f - p.z) / HALL_SUN.z;
    if (s0 > 0.0) {
        vec3 a = p + HALL_SUN * s0;
        float w = 0.003 + s0 * disc;
        m *= 1.0 - smoothstep(-0.5 * w, 0.5 * w, hallOpening(a.xy));
    }
    float s1 = (%.6f - p.z) / HALL_SUN.z;
    if (s1 > 0.0) {
        vec3 a = p + HALL_SUN * s1;
        float w = 0.003 + s1 * disc;
        m *= 1.0 - smoothstep(-0.5 * w, 0.5 * w, hallOpening(a.xy));
        float sm = 0.5 * (s0 + s1);
        vec3 c = p + HALL_SUN * sm;
        m *= hallBars(c.xy, 0.003 + max(sm, 0.0) * disc);
    }
    return m;
}
)",
                      h.toSun.x, h.toSun.y, h.toSun.z,
                      h.firstX, h.bay, static_cast<float>(h.windows - 1), h.firstX, h.bay, h.sill, 0.5f * h.width, h.spring - h.sill, h.spring - h.sill, 0.5f * h.width,
                      h.firstX, h.bay, static_cast<float>(h.windows - 1), h.firstX, h.bay, h.sill, h.mullion(), h.transom(0), h.transom(1), h.transom(2), h.bar,
                      h.zN, h.zN - h.wall);
        return b;
    }

    // Lets a standard material's shader know the hall's sun: the first directional light (the one that casts
    // shadows, which three.js sorts first) is let through only where the windows let it.
    inline void addSun(threepp::Shader& s, const std::string& glsl) {
        const auto replace = [](std::string& src, const std::string& what, const std::string& with) {
            const auto at = src.find(what);
            if (at != std::string::npos) src.replace(at, what.size(), with);
        };
        replace(s.vertexShader, "#include <common>", "#include <common>\nvarying vec3 vHallWorld;");
        replace(s.vertexShader, "#include <worldpos_vertex>", R"(#include <worldpos_vertex>
{
    vec4 hallWorld = vec4(transformed, 1.0);
    #ifdef USE_INSTANCING
    hallWorld = instanceMatrix * hallWorld;
    #endif
    vHallWorld = (modelMatrix * hallWorld).xyz;
})");
        replace(s.fragmentShader, "#include <common>", "#include <common>\nvarying vec3 vHallWorld;\n" + glsl);
        std::string lights = threepp::shaders::ShaderChunk::instance().lights_fragment_begin();
        replace(lights, "getDirectionalDirectLightIrradiance( directionalLight, geometry, directLight );",
                "getDirectionalDirectLightIrradiance( directionalLight, geometry, directLight );\n"
                "\t\t#if ( UNROLLED_LOOP_INDEX == 0 )\n"
                "\t\tdirectLight.color *= hallSun( vHallWorld );\n"
                "\t\t#endif\n");
        replace(s.fragmentShader, "#include <lights_fragment_begin>", lights);
    }

    // A standard material in the hall's light, with an optional pattern of its own in world space.
    inline std::shared_ptr<threepp::MeshStandardMaterial> hallMaterial(const std::string& glsl, const std::string& name, const threepp::Color& color,
                                                                      float roughness, const std::string& colorCode = "",
                                                                      const std::string& roughnessCode = "", const std::string& functions = "") {
        using namespace threepp;
        auto m = MeshStandardMaterial::create();
        m->color = color;
        m->roughness = roughness;
        m->metalness = 0.f;
        m->customProgramCacheKey = "plank-cathedral-hall-" + name;
        m->onBeforeCompile = [glsl, colorCode, roughnessCode, functions](Shader& s) {
            addSun(s, glsl);
            const auto replace = [](std::string& src, const std::string& what, const std::string& with) {
                const auto at = src.find(what);
                if (at != std::string::npos) src.replace(at, what.size(), with);
            };
            replace(s.fragmentShader, "varying vec3 vHallWorld;", "varying vec3 vHallWorld;\n" + functions);
            if (!colorCode.empty()) replace(s.fragmentShader, "#include <color_fragment>", "#include <color_fragment>\n" + colorCode);
            if (!roughnessCode.empty()) replace(s.fragmentShader, "#include <roughnessmap_fragment>", "#include <roughnessmap_fragment>\n" + roughnessCode);
        };
        return m;
    }

    inline const char* hallNoise() {
        return R"(
float hallHash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float hallNoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hallHash(i), hallHash(i + vec2(1, 0)), f.x), mix(hallHash(i + vec2(0, 1)), hallHash(i + vec2(1, 1)), f.x), f.y);
}
float hallMottle(vec2 p) {
    return 0.5 * hallNoise(p * 0.7) + 0.3 * hallNoise(p * 2.3 + 7.1) + 0.2 * hallNoise(p * 9.0 + 3.7);
}
// herringbone of blocks 1 x n (p in block widths): the block's id, then along and across it
vec4 herringbone(vec2 p, float n) {
    float k = floor(p.y);
    float c0 = p.x - k + n;
    float m = floor(c0 / (2.0 * n));
    float c = c0 - 2.0 * n * m;
    if (c >= n) return vec4(m, k, c - n, fract(p.y));
    float j = k - n + floor(c);
    return vec4(m + 0.5, j, p.y - (j + 1.0), fract(c));
}
)";
    }

    // Old oak parquet, laid in herringbone: each block a little different, the joints hairlines, the varnish worn
    // a little unevenly. Quiet: from where the cameras stand it reads as a wooden floor.
    inline std::shared_ptr<threepp::MeshStandardMaterial> parquetMaterial(const std::string& glsl) {
        using namespace threepp;
        Color c;
        c.setRGB(0.43f, 0.355f, 0.28f, ColorSpace::sRGB);
        return hallMaterial(glsl, "parquet", c, 0.42f,
                            R"(
{
    const float W = 0.07;
    vec2 q = vec2(vHallWorld.x + vHallWorld.z, vHallWorld.x - vHallWorld.z) * 0.70710678 / W;
    vec4 hb = herringbone(q, 5.0);
    float h0 = hallHash(hb.xy + 0.37), h1 = hallHash(hb.xy * 1.7 + 4.1), h2 = hallHash(hb.xy * 0.61 + 9.3);
    // where a block is only a few pixels across, its pattern would alias: it fades to the floor's average
    float near = 1.0 - smoothstep(0.12, 0.45, max(fwidth(q.x), fwidth(q.y)));
    // each block its own tint, a little lighter or darker, a little warmer or cooler
    vec3 tint = vec3(1.0 + 0.03 * (h1 - 0.5), 1.0, 1.0 - 0.04 * (h1 - 0.5)) * (0.965 + 0.05 * h0);
    // fibres along the block
    float fib = hallNoise(vec2(hb.z * 1.5 + h2 * 13.0, hb.w * 26.0 + h0 * 31.0));
    tint *= 0.98 + 0.04 * fib;
    // the joints: hairlines between blocks
    float aa = 1.5 * fwidth(hb.w) + 0.004;
    float seam = smoothstep(0.0, aa, min(hb.w, 1.0 - hb.w)) * smoothstep(0.0, aa, min(hb.z, 5.0 - hb.z));
    tint *= mix(0.91, 1.0, seam);
    tint = mix(vec3(0.98), tint, near);
    // the whole floor's patina, over metres
    tint *= 0.94 + 0.10 * hallMottle(vHallWorld.xz * 0.6);
    diffuseColor.rgb *= tint;
})",
                            R"(roughnessFactor = clamp(roughnessFactor + 0.08 * (hallHash(herringbone(vec2(vHallWorld.x + vHallWorld.z, vHallWorld.x - vHallWorld.z) * 0.70710678 / 0.07, 5.0).xy) - 0.5) + 0.14 * (hallMottle(vHallWorld.xz * 1.3 + 11.0) - 0.5), 0.0, 1.0);)",
                            hallNoise());
    }

    // Warm off-white lime plaster, its value drifting a little over metres.
    inline std::shared_ptr<threepp::MeshStandardMaterial> plasterMaterial(const std::string& glsl) {
        using namespace threepp;
        Color c;
        c.setRGB(0.84f, 0.82f, 0.77f, ColorSpace::sRGB);
        return hallMaterial(glsl, "plaster", c, 0.92f,
                            R"(diffuseColor.rgb *= 0.95 + 0.07 * hallMottle(vec2(vHallWorld.x + vHallWorld.z, vHallWorld.y) * 0.45);
// the skirting, dark oiled wood (drawn here rather than as boards: no thin slab to fight the wall in the depth buffer)
float hallBase = smoothstep(0.215, 0.215 + 1.5 * fwidth(vHallWorld.y) + 0.001, vHallWorld.y);
diffuseColor.rgb = mix(vec3(0.17, 0.125, 0.09), diffuseColor.rgb, hallBase);)",
                            "roughnessFactor = mix(0.5, roughnessFactor, smoothstep(0.215, 0.22, vHallWorld.y));", hallNoise());
    }

    // The interior of the hall, added to `scene`.
    inline void buildHall(threepp::Scene& scene, const Hall& h, const std::string& glsl) {
        using namespace threepp;
        const float cx = 0.5f * (h.x0 + h.x1), cz = 0.5f * (h.zN + h.zS);
        const float lengthX = h.x1 - h.x0, depthZ = h.zS - h.zN;

        auto plaster = plasterMaterial(glsl);
        Color frameColor;
        frameColor.setRGB(0.30f, 0.29f, 0.27f, ColorSpace::sRGB);
        auto frame = hallMaterial(glsl, "frame", frameColor, 0.6f);
        frame->side = Side::Double;// the arches' rings are flat
        Color ceilingColor;
        ceilingColor.setRGB(0.20f, 0.15f, 0.11f, ColorSpace::sRGB);
        auto ceiling = hallMaterial(glsl, "ceiling", ceilingColor, 0.85f);

        const auto box = [&](float x, float y, float z, float sx, float sy, float sz, const std::shared_ptr<Material>& m) {
            auto mesh = Mesh::create(BoxGeometry::create(sx, sy, sz), m);
            mesh->position.set(x, y, z);
            mesh->receiveShadow = true;
            scene.add(mesh);
            return mesh;
        };

        // the floor
        {
            auto floor = Mesh::create(PlaneGeometry::create(lengthX + 1, depthZ + 1), parquetMaterial(glsl));
            floor->rotation.x = -math::PI / 2;
            floor->position.set(cx, 0, cz);
            floor->receiveShadow = true;
            scene.add(floor);
        }

        // the north wall: a slab with the arched openings cut through it
        {
            Shape wall;
            wall.moveTo(h.x0 - h.wall, -0.5f);
            wall.lineTo(h.x1 + h.wall, -0.5f);
            wall.lineTo(h.x1 + h.wall, h.height + 0.5f);
            wall.lineTo(h.x0 - h.wall, h.height + 0.5f);
            wall.lineTo(h.x0 - h.wall, -0.5f);
            const float hw = 0.5f * h.width;
            for (int k = 0; k < h.windows; ++k) {
                const float x = h.windowX(k);
                auto hole = std::make_shared<Path>();
                hole->moveTo(x - hw, h.sill);
                hole->lineTo(x + hw, h.sill);
                hole->lineTo(x + hw, h.spring);
                hole->absarc(x, h.spring, hw, 0, math::PI, false);
                hole->lineTo(x - hw, h.sill);
                wall.holes.push_back(hole);
            }
            ExtrudeGeometry::Options o;
            o.depth = h.wall;
            o.bevelEnabled = false;
            o.curveSegments = 40;
            auto mesh = Mesh::create(ExtrudeGeometry::create({wall}, o), plaster);
            mesh->position.set(0, 0, h.zN - h.wall);
            scene.add(mesh);

            // the frames: a mullion either side of the middle up into the arch, three transoms
            const float zf = h.zN - 0.5f * h.wall;
            for (int k = 0; k < h.windows; ++k) {
                const float x = h.windowX(k);
                for (const float side : {-1.f, 1.f}) {
                    const float u = side * h.mullion();
                    const float top = h.spring + std::sqrt(std::max(0.f, hw * hw - u * u));
                    box(x + u, 0.5f * (h.sill + top), zf, 2 * h.bar, top - h.sill, 0.07f, frame);
                }
                for (int t = 0; t < 3; ++t) box(x, h.sill + h.transom(t), zf, h.width, 2 * h.bar, 0.07f, frame);
                // the frame round the opening's straight sides and its sill
                box(x - hw + h.bar + 0.004f, 0.5f * (h.sill + h.spring), zf, 2 * h.bar, h.spring - h.sill, 0.08f, frame);
                box(x + hw - h.bar - 0.004f, 0.5f * (h.sill + h.spring), zf, 2 * h.bar, h.spring - h.sill, 0.08f, frame);
                box(x, h.sill + h.bar + 0.004f, zf, h.width - 0.008f, 2 * h.bar, 0.08f, frame);
                auto arc = Mesh::create(RingGeometry::create(hw - 2 * h.bar, hw, 48, 1, 0, math::PI), frame);
                arc->position.set(x, h.spring, zf);
                scene.add(arc);
            }
            // pilasters on the piers between the windows and at the ends
            for (int k = 0; k <= h.windows; ++k) {
                const float x = k == 0 ? h.x0 + 0.45f : k == h.windows ? h.x1 - 0.45f : h.windowX(k) - 0.5f * h.bay;
                box(x, 0.5f * h.height, h.zN + 0.15f, 0.8f, h.height, 0.3f, plaster);
            }
        }

        // the other three walls, plain, with pilasters in the same rhythm
        box(cx, 0.5f * h.height, h.zS + 0.5f * h.wall, lengthX + 2 * h.wall, h.height + 1, h.wall, plaster);
        box(h.x0 - 0.5f * h.wall, 0.5f * h.height, cz, h.wall, h.height + 1, depthZ + 2 * h.wall, plaster);
        box(h.x1 + 0.5f * h.wall, 0.5f * h.height, cz, h.wall, h.height + 1, depthZ + 2 * h.wall, plaster);
        for (int k = 0; k <= h.windows; ++k) {
            const float x = k == 0 ? h.x0 + 0.45f : k == h.windows ? h.x1 - 0.45f : h.windowX(k) - 0.5f * h.bay;
            box(x, 0.5f * h.height, h.zS - 0.15f, 0.8f, h.height, 0.3f, plaster);
        }
        for (const float z : {h.zN + 5.f, h.zN + 10.f}) {
            box(h.x0 + 0.15f, 0.5f * h.height, z, 0.3f, h.height, 0.8f, plaster);
            box(h.x1 - 0.15f, 0.5f * h.height, z, 0.3f, h.height, 0.8f, plaster);
        }

        // the ceiling: dark timber, beams across from pilaster to pilaster
        {
            auto plane = Mesh::create(PlaneGeometry::create(lengthX + 1, depthZ + 1), ceiling);
            plane->rotation.x = math::PI / 2;
            plane->position.set(cx, h.height, cz);
            scene.add(plane);
            for (int k = 0; k <= h.windows; ++k) {
                const float x = k == 0 ? h.x0 + 0.45f : k == h.windows ? h.x1 - 0.45f : h.windowX(k) - 0.5f * h.bay;
                box(x, h.height - 0.3f, cz, 0.5f, 0.6f, depthZ, ceiling);
            }
            // a plain cornice where the walls meet it
            box(cx, h.height - 0.35f, h.zN + 0.2f, lengthX, 0.3f, 0.4f, plaster);
            box(cx, h.height - 0.35f, h.zS - 0.2f, lengthX, 0.3f, 0.4f, plaster);
        }

        // outside: the bright late afternoon beyond the windows, overexposed as it is seen from a dim room
        {
            auto material = ShaderMaterial::create();
            material->vertexShader = R"(
varying vec3 vOut;
void main() {
    vec4 world = modelMatrix * vec4(position, 1.0);
    vOut = world.xyz;
    gl_Position = projectionMatrix * viewMatrix * world;
})";
            char sky[1024];
            std::snprintf(sky, sizeof(sky), R"(
varying vec3 vOut;
void main() {
    // the late afternoon sky seen past the window's foot: paler and a little cooler high up, warmer toward the sun
    float up = smoothstep(3.0, 11.0, vOut.y);
    float sunward = 1.0 - smoothstep(%.3f, %.3f, vOut.x);
    vec3 c = mix(vec3(1.32, 1.20, 1.02), vec3(1.10, 1.13, 1.18), up);
    c *= mix(vec3(1.0), vec3(1.07, 1.0, 0.90), sunward);
    gl_FragColor = vec4(c, 1.0);
    #include <tonemapping_fragment>
    #include <encodings_fragment>
})",
                          h.x0, h.x1);
            material->fragmentShader = sky;
            material->fog = false;
            auto outside = Mesh::create(PlaneGeometry::create(lengthX + 30, h.height + 10), material);
            outside->position.set(cx, 0.5f * h.height, h.zN - h.wall - 3.f);
            scene.add(outside);
        }
    }

    // The light the room gives back, as an environment: a warm glow from the floor's patches of sun below, the dim
    // plaster all round, a dark ceiling, the bright windows along the north, and (`fill`) the broad warm light the
    // sunlit floor and walls throw back from the side away from the sun, low. As light from the environment it
    // lifts the shadow sides without the glare a directional fill leaves on the varnished floor.
    inline std::shared_ptr<threepp::Texture> hallEnvironment(threepp::GLRenderer& renderer, const Hall& h, float fill) {
        using namespace threepp;
        Scene room;
        auto material = ShaderMaterial::create();
        material->vertexShader = R"(
varying vec3 vDir;
void main() {
    vDir = normalize(position);
    gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(position, 1.0);
})";
        char b[1536];
        std::snprintf(b, sizeof(b), R"(
varying vec3 vDir;
void main() {
    vec3 d = normalize(vDir);
    vec3 floorC = vec3(0.21, 0.155, 0.10);
    vec3 wallC = vec3(0.078, 0.082, 0.09);
    vec3 ceilC = vec3(0.018, 0.018, 0.019);
    vec3 c = d.y < 0.0 ? mix(wallC, floorC, smoothstep(0.0, 0.4, -d.y)) : mix(wallC, ceilC, smoothstep(0.2, 0.8, d.y));
    // the windows: high along the north, brightest toward the sun
    float north = smoothstep(0.1, 0.8, -d.z);
    float band = smoothstep(0.25, 0.4, d.y) * (1.0 - smoothstep(0.7, 0.85, d.y));
    float sun = pow(max(0.0, dot(d, vec3(%.4f, %.4f, %.4f))), 4.0);
    c += band * (vec3(0.88, 0.93, 1.0) * 1.05 * north + vec3(1.0, 0.88, 0.72) * 0.9 * sun);
    // the room's bounce from the side away from the sun, low and broad
    vec3 away = normalize(vec3(%.4f, 0.15, %.4f));
    c += vec3(1.0, 0.82, 0.62) * %.4f * pow(max(0.0, dot(d, away)), 2.0);
    gl_FragColor = vec4(c, 1.0);
})",
                      h.toSun.x, h.toSun.y, h.toSun.z, -h.toSun.x, -h.toSun.z, fill);
        material->fragmentShader = b;
        material->side = Side::Back;
        material->depthWrite = false;
        auto sphere = Mesh::create(SphereGeometry::create(20, 64, 32), material);
        room.add(sphere);
        PMREMGenerator pmrem(renderer);
        return pmrem.fromScene(room, 0.02f, 0.1f, 50.f);
    }

}// namespace planks

#endif//PLANKCATHEDRAL_HALL_HPP
