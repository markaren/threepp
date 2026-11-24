// The stream's water and the flume's.
//
// The stream is simulated (ShallowWater.hpp: the shallow-water equations on a 12.5 cm grid) over the real bed: the
// ground, raised wherever a boulder stands (each boulder's own vertices rasterised into the grid), walled by the
// bridge's piers, the wheel's pier and the mill. Water comes in at the top of the grid, the wheel's tail water comes
// in under the wheel, and it leaves over the bottom edge. Its surface is one mesh whose vertices follow the simulation.
//
// The material is a MeshStandardMaterial (lit, shadowed and reflecting the sky as everything else is) given two
// things on GL by onBeforeCompile: the ripple map dragged along the simulated current (a flow map in two phases), and
// the water's body: the bed seen through it darkens and browns with depth (tea-coloured, as a forest stream is),
// and where the simulation says it is white, it is. The same material carries the Vulkan renderer's own flow mode
// (flow, flowTime, the velocity in "uv2", the foam in "uv").

#ifndef THREEPP_EXAMPLE_WATERMILL_WATER_HPP
#define THREEPP_EXAMPLE_WATERMILL_WATER_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/objects/Mesh.hpp"

#include "ShallowWater.hpp"
#include "Site.hpp"
#include "Textures.hpp"

#include <array>
#include <vector>

namespace mill {

    // ---- the water's material -----------------------------------------------------------------------------------------
    // GL draws the water in two passes over the same mesh. The first (makeWaterTint) multiplies what is behind by the
    // light that gets through: the bed seen down through the water and back up, absorbed per channel (tea brown: blue
    // goes first) and less what the surface reflects. The second (this) adds what the surface sends back: the
    // reflection (a planar mirror of the scene where it holds something, the environment elsewhere), the light the
    // water's body scatters, and white water, which covers what is behind it.
    inline std::shared_ptr<threepp::MeshPhysicalMaterial> makeWaterMaterial(const std::shared_ptr<threepp::Texture>& ripples) {
        using namespace threepp;
        auto m = MeshPhysicalMaterial::create();
        m->color.setRGB(0.014f, 0.0095f, 0.0045f);// what the water scatters back: little, and tannin brown
        m->roughness = 0.05f;
        m->metalness = 0.f;
        m->transparent = true;
        m->blending = Blending::Custom;// premultiplied: the reflection adds, the bed shows through by (1 - alpha)
        m->blendSrc = BlendFactor::One;
        m->blendDst = BlendFactor::OneMinusSrcAlpha;
        m->depthWrite = true;
        // the Vulkan renderer's flow mode (GL ignores these)
        m->flow = 1.f;
        m->flowTile = 0.6f;
        m->flowPeriod = 0.8f;
        m->flowFoam = 1.f;
        m->flowFoamAlbedo = 0.78f;
        m->customUniforms["uRipples"] = Uniform(static_cast<Texture*>(ripples.get()));
        m->customUniforms["uTime"] = Uniform(0.f);
        m->customUniforms["uTile"] = Uniform(0.6f);
        m->customUniforms["uPeriod"] = Uniform(0.8f);
        m->customUniforms["uAbsorb"] = Uniform(Vector3(0.45f, 0.9f, 1.9f));
        m->customProgramCacheKey = "watermill-water";
        m->onBeforeCompile = [](Shader& s) {
            const auto replace = [](std::string& src, const std::string& what, const std::string& with) {
                const auto at = src.find(what);
                if (at != std::string::npos) src.replace(at, what.size(), with);
            };
            replace(s.vertexShader, "#include <common>", R"(#include <common>
attribute vec2 flowVel;
attribute vec2 waterInfo;
varying vec2 vFlowVel;
varying vec2 vWaterInfo;
varying vec3 vWaterWorld;)");
            replace(s.vertexShader, "#include <begin_vertex>", R"(#include <begin_vertex>
vFlowVel = flowVel;
vWaterInfo = waterInfo;
vWaterWorld = (modelMatrix * vec4(transformed, 1.0)).xyz;)");
            replace(s.fragmentShader, "#include <common>", R"(#include <common>
uniform sampler2D uRipples;
uniform float uTime;
uniform float uTile;
uniform float uPeriod;
uniform vec3 uAbsorb;
varying vec2 vFlowVel;
varying vec2 vWaterInfo;
varying vec3 vWaterWorld;)");
            replace(s.fragmentShader, "#include <normal_fragment_begin>", R"(#include <normal_fragment_begin>
// the ripples, carried along the current in two phases half a period apart
float wPh0 = fract(uTime / uPeriod);
float wPh1 = fract(uTime / uPeriod + 0.5);
float wBlend = abs(1.0 - 2.0 * wPh0);
vec2 wBase = vWaterWorld.xz / uTile;
vec4 wT0 = texture2D(uRipples, wBase - vFlowVel * (wPh0 * uPeriod / uTile));
vec4 wT1 = texture2D(uRipples, wBase - vFlowVel * (wPh1 * uPeriod / uTile) + vec2(0.37, 0.61));
vec4 waterTap = mix(wT0, wT1, wBlend);
// a finer, slower layer: even the stillest water shimmers a little
vec4 wT2 = texture2D(uRipples, vWaterWorld.xz / (uTile * 0.31) + vec2(uTime * 0.021, -uTime * 0.017));
float wSpeed = length(vFlowVel);
// the flow's own frame: streaks of white water and the chop's crests are drawn out along the current
vec2 wDir = wSpeed > 1e-3 ? vFlowVel / wSpeed : vec2(0.0, 1.0);
vec2 wAcross = vec2(-wDir.y, wDir.x);
vec2 wStreakUv = vec2(dot(vWaterWorld.xz, wAcross) / (uTile * 1.1), dot(vWaterWorld.xz, wDir) / (uTile * 4.5));
float wRun = wSpeed * uPeriod / (uTile * 4.5);
float wStreak = mix(texture2D(uRipples, wStreakUv - vec2(0.0, wPh0 * wRun)).a, texture2D(uRipples, wStreakUv - vec2(0.0, wPh1 * wRun) + vec2(0.41, 0.17)).a, wBlend);
// running water is choppy all over: a third, small and fast layer carried with it
vec4 wT3 = texture2D(uRipples, wStreakUv * vec2(2.3, 6.0) - vec2(0.0, uTime * wSpeed / (uTile * 0.6)));
float wChop = clamp(wSpeed / 0.8, 0.0, 1.0);
vec2 wTilt = (waterTap.xy * 2.0 - 1.0) * (0.24 + 0.86 * wChop) + (wT2.xy * 2.0 - 1.0) * 0.12 + (wT3.xy * 2.0 - 1.0) * 0.32 * wChop;
vec3 wN = normalize((vec4(normal, 0.0) * viewMatrix).xyz);
vec3 wT = normalize(vec3(1.0, 0.0, 0.0) - wN * wN.x);
vec3 wB = normalize(vec3(0.0, 0.0, 1.0) - wN * wN.z);
vec3 waterNormal = normalize(wT * wTilt.x + wB * wTilt.y + wN * sqrt(max(1e-4, 1.0 - dot(wTilt, wTilt))));
normal = normalize((viewMatrix * vec4(waterNormal, 0.0)).xyz);)");
            replace(s.fragmentShader, "#include <opaque_fragment>", R"(// the body: the bed seen down through the water and back up, browned and darkened with depth
float wDepth = vWaterInfo.y;
vec3 wView = normalize(cameraPosition - vWaterWorld);
float wCos = max(dot(wView, waterNormal), 0.05);
vec3 wThrough = exp(-uAbsorb * wDepth * (1.0 + 1.0 / max(wCos, 0.25)));
float wSeen = dot(wThrough, vec3(1.0 / 3.0));
float wEdge = smoothstep(0.0, 0.03, wDepth);
vec3 wIrr = totalDiffuse / max(diffuseColor.rgb, vec3(1e-3));
outgoingLight = totalDiffuse * (1.0 - wSeen) + totalSpecular;
float wAlpha = 0.0;// the bed's dimming is the tint pass's
// white water where the simulation says so, broken up by the lace carried with the ripples
float wFoam = vWaterInfo.x;
float wTh = 0.95 - 0.55 * wFoam;
// in fast water the white is in streaks along the current, in slow water in a lace
float wLace = mix(waterTap.a, wStreak, smoothstep(0.35, 1.1, wSpeed));
float wWhite = max(smoothstep(wTh - 0.1, wTh + 0.1, wLace), 0.55 * smoothstep(0.7, 1.0, wFoam)) * smoothstep(0.03, 0.25, wFoam);
outgoingLight = mix(outgoingLight, wIrr * 0.58, wWhite);
wAlpha = mix(wAlpha, 1.0, 0.9 * wWhite);
outgoingLight *= wEdge;
diffuseColor.a = wAlpha * wEdge;
#include <opaque_fragment>)");
        };
        return m;
    }

    // the first of GL's two water passes: what is behind the water times the light that comes through it
    inline std::shared_ptr<threepp::ShaderMaterial> makeWaterTint() {
        using namespace threepp;
        auto m = ShaderMaterial::create();
        m->vertexShader = R"(
attribute vec2 waterInfo;
varying vec3 vWorld;
varying vec3 vNormalW;
varying vec2 vInfo;
void main() {
    vec4 world = modelMatrix * vec4(position, 1.0);
    vWorld = world.xyz;
    vNormalW = normalize(mat3(modelMatrix) * normal);
    vInfo = waterInfo;
    gl_Position = projectionMatrix * viewMatrix * world;
})";
        m->fragmentShader = R"(
uniform vec3 uAbsorb;
varying vec3 vWorld;
varying vec3 vNormalW;
varying vec2 vInfo;
void main() {
    float depth = vInfo.y;
    vec3 V = normalize(cameraPosition - vWorld);
    float c = max(dot(normalize(vNormalW), V), 0.05);
    float F = 0.02 + 0.98 * pow(1.0 - c, 5.0);
    vec3 T = exp(-uAbsorb * depth * (1.0 + 1.0 / max(c, 0.25))) * (1.0 - F);
    gl_FragColor = vec4(mix(vec3(1.0), T, smoothstep(0.0, 0.03, depth)), 1.0);
})";
        m->uniforms = {{"uAbsorb", Uniform(Vector3(0.45f, 0.9f, 1.9f))}};
        m->transparent = true;
        m->depthWrite = false;
        m->blending = Blending::Custom;
        m->blendSrc = BlendFactor::Zero;
        m->blendDst = BlendFactor::SrcColor;
        return m;
    }

    // The same water for the Vulkan renderer and its path tracer: a real interface (ior 1.333) the light refracts
    // through, absorbing toward tea brown over its depth (Beer-Lambert), with the ripples as a normal map carried
    // along the simulated current by the renderer's flow mode.
    inline std::shared_ptr<threepp::MeshPhysicalMaterial> makeWaterMaterialPhysical(const std::shared_ptr<threepp::Texture>& ripples) {
        using namespace threepp;
        auto m = MeshPhysicalMaterial::create();
        m->color.setRGB(1.f, 1.f, 1.f);
        m->roughness = 0.03f;
        m->metalness = 0.f;
        m->transmission = 1.f;
        m->ior = 1.333f;
        m->thickness = 0.3f;
        m->attenuationColor = Color(0.66f, 0.46f, 0.25f);
        m->attenuationDistance = 0.45f;
        m->normalMap = ripples;
        m->normalScale.set(1.1f, 1.1f);
        m->flow = 1.f;
        m->flowTile = 0.6f;
        m->flowPeriod = 0.8f;
        m->flowFoam = 1.f;
        m->flowFoamAlbedo = 0.78f;
        return m;
    }

    // ---- the stream -------------------------------------------------------------------------------------------------
    class StreamWater {
    public:
        static constexpr float dx = 0.125f, x0 = -16.f, z0 = -42.f;
        static constexpr int nx = 204, nz = 784;// to x 9.5, z 56 (past the falls and their pool)
        static constexpr float inflow = 2.8f;   // m3/s in at the top (early summer: good water, as the photos show it)

        ShallowWater swe{nx, nz, dx, x0, z0};
        std::shared_ptr<threepp::Mesh> mesh;
        std::shared_ptr<threepp::Mesh> tint;   // GL's absorption pass over the same surface (null without a material)
        std::shared_ptr<threepp::Mesh> curtain;// the falls: the sheet that goes over the ledge
        // White water for VulkanRenderer and its path tracer, whose surface does not draw the flow mode's foam: a thin
        // skin over the water, white and rough, as opaque as the simulation's foam (null unless asked for)
        std::shared_ptr<threepp::Mesh> foamSkin;

        void addFoamSkin() {
            using namespace threepp;
            auto m = MeshStandardMaterial::create();
            m->color.setRGB(0.78f, 0.78f, 0.76f);
            m->map = foamLaceTexture(611);// its alpha cuts the lace
            m->alphaTest = 0.5f;
            m->roughness = 0.85f;
            m->metalness = 0.f;
            m->vertexColors = true;
            const size_t n = static_cast<size_t>(nx) * nz;
            skinPos_.assign(3 * n, 0.f);
            skinCol_.assign(4 * n, 1.f);
            auto g = BufferGeometry::create();
            g->setAttribute("position", FloatBufferAttribute::create(skinPos_, 3));
            g->setAttribute("normal", FloatBufferAttribute::create(normals_, 3));
            g->setAttribute("color", FloatBufferAttribute::create(skinCol_, 4));
            std::vector<float> uv(2 * n);
            for (size_t c = 0; c < n; ++c) {
                uv[2 * c] = swe.xOf(static_cast<int>(c % nx)) / 1.2f;
                uv[2 * c + 1] = swe.zOf(static_cast<int>(c / nx)) / 1.2f;
            }
            g->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
            for (const auto& name : {"position", "normal", "color"}) g->getAttribute(name)->setUsage(DrawUsage::Dynamic);
            surfaceIndex_ = mesh->geometry()->getIndex()->array();
            g->setIndex(std::vector<unsigned int>{0, 0, 0});
            foamSkin = Mesh::create(g, m);
            foamSkin->frustumCulled = false;
            foamSkin->castShadow = false;
            foamSkin->receiveShadow = true;
            foamSkin->renderOrder = 3;
        }

        // rockPoints: every boulder's vertices (x, y, z), so the water flows round the boulders
        StreamWater(const std::vector<float>& rockPoints, const std::shared_ptr<threepp::Material>& material,
                    const std::shared_ptr<threepp::Material>& tintMaterial, const std::shared_ptr<threepp::Material>& fallMaterial) {
            const int n = nx * nz;
            std::vector<float> ground(n), rockTop(n, -1e9f), bed(n), cover(n);
            for (int j = 0; j < nz; ++j)
                for (int i = 0; i < nx; ++i) ground[j * nx + i] = site::groundHeight(swe.xOf(i), swe.zOf(j));
            for (size_t k = 0; k + 2 < rockPoints.size(); k += 3) {
                const int c = swe.cellAt(rockPoints[k], rockPoints[k + 2]);
                if (c >= 0) rockTop[c] = std::max(rockTop[c], rockPoints[k + 1]);
            }
            // a boulder fills its cells to its top (the highest of its vertices in the cell); the water's surface is carried
            // flat into the cells next to it, where the boulder hides it
            for (int c = 0; c < n; ++c) {
                bed[c] = std::max(ground[c], rockTop[c] - 0.01f);
                cover[c] = std::max(ground[c], rockTop[c]);
            }
            std::vector<uint8_t> active(n, 0);
            for (int j = 1; j < nz - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    const float x = swe.xOf(i), z = swe.zOf(j);
                    const int c = j * nx + i;
                    float cx, hw;
                    site::streamAt(z, cx, hw);
                    // the channel and a margin for the water to rise into, never out to the grid's edges
                    const float hs = site::bankHalfWidth(z, hw, x < cx ? -1.f : 1.f);
                    bool on = std::abs(x - cx) < hs + 2.2f && ground[c] < site::nominalWater(z) + 0.7f;
                    if (blocked(x, z)) on = false;
                    if (site::inWheelPit(x, z) && !site::inMill(x, z)) on = ground[c] < site::nominalWater(z) + 0.7f;
                    active[c] = on;
                }
            }
            // a rocky stream is white wherever it drops or runs fast between stones
            swe.foamSlope = 1.6f;
            swe.foamSlopeMin = 0.14f;
            swe.foamChute = 1.0f;
            swe.foamChuteSpeed = 0.55f;
            swe.foamPiling = 0.22f;
            swe.foamLife = 3.5f;
            swe.setBed(bed, ground, cover);
            swe.setActive(active);
            // the inflow along the top rows, the outflow held at the bottom
            int top = 0;
            for (int pass = 0; pass < 2; ++pass) {
                for (int j = 1; j < 5; ++j) {
                    for (int i = 0; i < nx; ++i) {
                        const int c = j * nx + i;
                        if (!active[c] || bed[c] > site::nominalWater(swe.zOf(j))) continue;
                        if (pass == 0) ++top;
                        else swe.setSource(c, inflow / (static_cast<float>(top) * dx * dx));
                    }
                }
            }
            for (int j = nz - 3; j < nz - 1; ++j)
                for (int i = 0; i < nx; ++i)
                    if (active[j * nx + i]) swe.setSink(j * nx + i, site::nominalWater(swe.zOf(j)) - 0.05f);
            // the tail water comes in under the wheel
            for (int j = 0; j < nz; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const float x = swe.xOf(i), z = swe.zOf(j);
                    if (active[j * nx + i] && std::abs(x - site::wheelX) < 0.45f && z > site::wheelZ + 0.6f && z < site::wheelZ + 1.8f) tail_.push_back(j * nx + i);
                }
            }
            for (int c = 0; c < n; ++c) {
                if (!active[c]) continue;
                const float level = site::nominalWater(swe.zOf(c / nx)) - 0.08f;
                if (level > bed[c]) swe.setDepth(c, level - bed[c]);
            }
            setTailInflow(0.1f);
            swe.settle(30.f);

            // the surface mesh: a vertex per cell; quads wherever water could be
            positions_.resize(3 * n);
            normals_.resize(3 * n);
            velocity_.resize(2 * n);
            info_.resize(2 * n);
            uv_.assign(2 * n, 0.f);
            swe.surface(positions_.data(), normals_.data(), velocity_.data(), info_.data());
            auto g = threepp::BufferGeometry::create();
            g->setAttribute("position", threepp::FloatBufferAttribute::create(positions_, 3));
            g->setAttribute("normal", threepp::FloatBufferAttribute::create(normals_, 3));
            g->setAttribute("flowVel", threepp::FloatBufferAttribute::create(velocity_, 2));
            g->setAttribute("waterInfo", threepp::FloatBufferAttribute::create(info_, 2));
            g->setAttribute("uv", threepp::FloatBufferAttribute::create(uv_, 2));
            g->setAttribute("uv2", threepp::FloatBufferAttribute::create(velocity_, 2));
            for (const auto& name : {"position", "normal", "flowVel", "waterInfo", "uv", "uv2"}) g->getAttribute(name)->setUsage(threepp::DrawUsage::Dynamic);
            std::vector<unsigned int> index;
            for (int j = 0; j + 1 < nz; ++j) {
                const float zq0 = swe.zOf(j), zq1 = swe.zOf(j + 1);
                for (int i = 0; i + 1 < nx; ++i) {
                    // over the falls' drop the curtain is the water, not a ramp of the surface
                    const float la = site::fallsLipZ(swe.xOf(i)), lb = site::fallsLipZ(swe.xOf(i + 1));
                    if (zq1 > std::min(la, lb) - 0.2f && zq0 < std::max(la, lb) + 0.45f) continue;
                    const unsigned a = j * nx + i, b = a + 1, c = a + nx, d = c + 1;
                    if (active[a] || active[b] || active[c] || active[d]) index.insert(index.end(), {a, c, b, b, c, d});
                }
            }
            g->setIndex(index);
            mesh = threepp::Mesh::create(g, material);
            mesh->frustumCulled = false;
            mesh->receiveShadow = true;
            mesh->renderOrder = 2;
            if (tintMaterial) {
                tint = threepp::Mesh::create(g, tintMaterial);
                tint->frustumCulled = false;
                tint->renderOrder = 1;
            }
            material_ = std::dynamic_pointer_cast<threepp::MeshStandardMaterial>(material);
            buildCurtain(fallMaterial);
            buildWaveFields();
        }

        // the falls this frame: per column of the lip, how much goes over (m2/s), where from and where it lands
        struct Pour {
            float x, top, bottom, speed, q;
        };
        std::vector<Pour> pours;

        // what the wheel lets fall into its pit, m3/s
        void setTailInflow(float q) {
            if (tail_.empty()) return;
            const float rate = q / (static_cast<float>(tail_.size()) * dx * dx);
            for (int c : tail_) swe.setSourceRate(c, rate);
        }

        void step(float dt) { swe.step(dt); }

        void updateCurtain() {
            using namespace site;
            pours.resize(nx);
            zLip_.resize(nx);
            for (int i = 0; i < nx; ++i) {
                const float x = swe.xOf(i);
                const float lipZ = fallsLipZ(x);
                const int jLip = std::clamp(static_cast<int>((lipZ - 0.35f - z0) / dx), 0, nz - 1);
                zLip_[i] = lipZ - 0.05f;
                const int c = jLip * nx + i;
                float vx, vz;
                swe.velocityAt(x, swe.zOf(jLip) + 0.5f * dx, vx, vz);
                const float h = swe.activeAt(c) ? swe.depth(c) : 0.f;
                const float q = h > 0.01f ? h * std::max(vz, 0.f) : 0.f;
                const float top = swe.bedAt(c) + h;
                float bottom = swe.surfaceAt(x, lipZ + 1.2f);
                if (bottom < -1e8f) bottom = groundHeight(x, lipZ + 1.2f);
                pours[i] = {x, top, bottom, std::max(vz, 0.4f), q};
            }
            // the sheet: a column per lip cell, falling from the lip along its path to the pool
            const int along = CurtainRows;
            for (int i = 0; i < nx; ++i) {
                const auto& p = pours[i];
                const float drop = std::max(p.top - p.bottom, 0.05f);
                const float T = std::sqrt(2.f * drop / 9.81f);
                // a trickle is a thread, a sheet is a sheet
                // a trickle is a thread, a sheet is a sheet: the strip narrows with the flow, it does not fade
                const float strength = 0.55f + 0.45f * smooth(0.03f, 0.12f, p.q);
                const float widthShare = 0.2f + 0.8f * smooth(0.02f, 0.11f, p.q);
                if (p.q < 0.012f) {
                    // no water over this cell: collapse its strip
                    for (int k = 0; k <= along; ++k)
                        for (int e = 0; e < 2; ++e) cCol_[4 * (static_cast<size_t>(k) * (2 * nx) + 2 * i + e) + 3] = 0.f;
                    continue;
                }
                const float zLip = zLip_[i];
                float dist = 0, py = p.top, pz = zLip;
                for (int k = 0; k <= along; ++k) {
                    const float t = T * static_cast<float>(k) / static_cast<float>(along);
                    const float y = p.top - 4.905f * t * t, z = zLip + p.speed * t;
                    if (k > 0) dist += std::hypot(y - py, z - pz);
                    py = y;
                    pz = z;
                    float ny = p.speed, nz = 9.81f * t;// the sheet's normal faces up and out
                    const float len = std::max(std::hypot(ny, nz), 1e-4f);
                    // each lip cell falls as its own strip (a little wider than the cell, so neighbours close up)
                    for (int e = 0; e < 2; ++e) {
                        const size_t v = static_cast<size_t>(k) * (2 * nx) + 2 * i + e;
                        // the strip pulls in as it falls, the more so where it is thin, and wavers
                        const float fall = static_cast<float>(k) / static_cast<float>(along);
                        const float narrow = 1.f - 0.35f * fall * (1.f - smooth(0.05f, 0.2f, p.q));
                        const float x = p.x + (e ? 0.56f : -0.56f) * dx * widthShare * narrow + 0.03f * fall * std::sin(p.x * 7.f + static_cast<float>(k));
                        cPos_[3 * v] = x;
                        cPos_[3 * v + 1] = y;
                        cPos_[3 * v + 2] = z;
                        cNor_[3 * v] = 0;
                        cNor_[3 * v + 1] = ny / len;
                        cNor_[3 * v + 2] = nz / len;
                        cUv_[2 * v] = x / 0.7f;
                        cUv_[2 * v + 1] = dist;
                        cCol_[4 * v + 3] = strength * (1.f - smooth(0.75f, 1.f, static_cast<float>(k) / static_cast<float>(along)));
                    }
                }
            }
            auto& g = *curtain->geometry();
            g.getAttribute<float>("position")->array() = cPos_;
            g.getAttribute<float>("normal")->array() = cNor_;
            g.getAttribute<float>("uv")->array() = cUv_;
            g.getAttribute<float>("color")->array() = cCol_;
            for (const auto& name : {"position", "normal", "uv", "color"}) g.getAttribute(name)->needsUpdate();
        }

        void updateMesh(float time) {
            updateCurtain();
            swe.surface(positions_.data(), normals_.data(), velocity_.data(), info_.data());
            addStandingWaves(time);
            for (size_t c = 0; c < info_.size() / 2; ++c) uv_[2 * c] = info_[2 * c];
            auto& g = *mesh->geometry();
            g.getAttribute<float>("position")->array() = positions_;
            g.getAttribute<float>("normal")->array() = normals_;
            g.getAttribute<float>("flowVel")->array() = velocity_;
            g.getAttribute<float>("waterInfo")->array() = info_;
            g.getAttribute<float>("uv")->array() = uv_;
            g.getAttribute<float>("uv2")->array() = velocity_;
            for (const auto& name : {"position", "normal", "flowVel", "waterInfo", "uv", "uv2"}) g.getAttribute(name)->needsUpdate();
            if (foamSkin) {
                const size_t n = info_.size() / 2;
                for (size_t c = 0; c < n; ++c) {
                    skinPos_[3 * c] = positions_[3 * c];
                    skinPos_[3 * c + 1] = positions_[3 * c + 1] + 0.012f;
                    skinPos_[3 * c + 2] = positions_[3 * c + 2];
                    // as the GL water's foam: only the whitest water, broken up (here by the standing-wave field)
                    const float lace = info_[2 * c] + 0.3f * wave1_[c];
                    skinCol_[4 * c + 3] = info_[2 * c + 1] > 0.f ? 0.85f * smooth(0.6f, 0.95f, lace) : 0.f;
                }
                // VulkanRenderer takes no alpha from vertex colours: the skin is there only where the foam is thick
                skinIndex_.clear();
                for (size_t t = 0; t + 2 < surfaceIndex_.size(); t += 3) {
                    const unsigned a = surfaceIndex_[t], b = surfaceIndex_[t + 1], c = surfaceIndex_[t + 2];
                    if (skinCol_[4 * a + 3] + skinCol_[4 * b + 3] + skinCol_[4 * c + 3] > 0.9f) skinIndex_.insert(skinIndex_.end(), {a, b, c});
                }
                if (skinIndex_.empty()) skinIndex_ = {0, 0, 0};
                auto& sg = *foamSkin->geometry();
                sg.setIndex(skinIndex_);
                sg.getAttribute<float>("position")->array() = skinPos_;
                sg.getAttribute<float>("normal")->array() = normals_;
                sg.getAttribute<float>("color")->array() = skinCol_;
                for (const auto& name : {"position", "normal", "color"}) sg.getAttribute(name)->needsUpdate();
            }
            if (material_) {
                material_->customUniforms["uTime"].setValue(time);
                material_->flowTime = time;
            }
        }

        // the water's level at a point, or a very low number where it is dry
        [[nodiscard]] float levelAt(float x, float z) const { return swe.surfaceAt(x, z); }

    private:
        // Standing waves and boils. Where water runs fast and shallow over a rough bed its surface stands up in waves
        // that hold their place over the rocks while the water runs through them (their wavelength grows with the
        // speed, about 2 pi v^2 / g), and boils that well up and sink. The simulation's 12.5 cm cells smooth these
        // out, so they are put back on its surface: a fixed field of bumps (two of them, mixed by a slow clock so the
        // waves breathe), scaled by the water's own velocity head v^2 / 2g and kept within its depth. Still water
        // gets none.
        std::vector<float> wave1_, wave2_, wg1x_, wg1z_, wg2x_, wg2z_;

        void buildWaveFields() {
            const int n = nx * nz;
            for (auto* v : {&wave1_, &wave2_, &wg1x_, &wg1z_, &wg2x_, &wg2z_}) v->assign(n, 0.f);
            forRows(nz, [&](int j) {
                for (int i = 0; i < nx; ++i) {
                    const float x = swe.xOf(i), z = swe.zOf(j);
                    // stretched across the flow (the creek runs along z): crests lie across it
                    wave1_[j * nx + i] = fbm2(x * 0.9f, z * 1.5f, 3, 501u);
                    wave2_[j * nx + i] = fbm2(x * 1.3f + 7.f, z * 1.1f, 3, 502u);
                }
            });
            forRows(nz, [&](int j) {
                const int jb = std::max(j - 1, 0), jt = std::min(j + 1, nz - 1);
                for (int i = 0; i < nx; ++i) {
                    const int il = std::max(i - 1, 0), ir = std::min(i + 1, nx - 1);
                    const int c = j * nx + i;
                    const float sx = 1.f / (static_cast<float>(ir - il) * dx), sz = 1.f / (static_cast<float>(jt - jb) * dx);
                    wg1x_[c] = (wave1_[j * nx + ir] - wave1_[j * nx + il]) * sx;
                    wg1z_[c] = (wave1_[jt * nx + i] - wave1_[jb * nx + i]) * sz;
                    wg2x_[c] = (wave2_[j * nx + ir] - wave2_[j * nx + il]) * sx;
                    wg2z_[c] = (wave2_[jt * nx + i] - wave2_[jb * nx + i]) * sz;
                }
            });
        }

        void addStandingWaves(float time) {
            const float c1 = std::cos(0.9f * time), s1 = std::sin(0.9f * time);
            forRows(nz, [&](int j) {
                for (int i = 0; i < nx; ++i) {
                    const int c = j * nx + i;
                    const float h = info_[2 * c + 1];
                    if (h <= 0.f) continue;
                    const float vx = velocity_[2 * c], vz = velocity_[2 * c + 1];
                    const float head = (vx * vx + vz * vz) / (2.f * 9.81f);
                    // a wave stands about as high as the velocity head, in shallow water no higher than half its depth
                    const float a = std::min({0.75f * head, 0.5f * h, 0.13f}) * smooth(0.015f, 0.06f, h);
                    if (a < 1e-4f) continue;
                    const float mixA = 0.75f + 0.25f * c1, mixB = 0.45f * s1;
                    positions_[3 * c + 1] += a * (mixA * wave1_[c] + mixB * wave2_[c]);
                    const float fx = a * (mixA * wg1x_[c] + mixB * wg2x_[c]), fz = a * (mixA * wg1z_[c] + mixB * wg2z_[c]);
                    float nxv = normals_[3 * c] - fx * normals_[3 * c + 1], nyv = normals_[3 * c + 1], nzv = normals_[3 * c + 2] - fz * normals_[3 * c + 1];
                    const float len = std::sqrt(nxv * nxv + nyv * nyv + nzv * nzv);
                    normals_[3 * c] = nxv / len;
                    normals_[3 * c + 1] = nyv / len;
                    normals_[3 * c + 2] = nzv / len;
                    // the crests break white
                    const float crest = smooth(0.35f, 0.7f, mixA * wave1_[c] + mixB * wave2_[c]) * smooth(0.04f, 0.1f, a);
                    info_[2 * c] = std::max(info_[2 * c], 0.55f * crest);
                }
            });
        }

        static constexpr int CurtainRows = 12;
        std::vector<float> zLip_;
        std::vector<float> skinPos_, skinCol_;
        std::vector<unsigned int> surfaceIndex_, skinIndex_;
        std::vector<float> cPos_, cNor_, cUv_, cCol_;

        void buildCurtain(const std::shared_ptr<threepp::Material>& material) {
            const int n = (CurtainRows + 1) * 2 * nx;
            cPos_.assign(3 * n, 0.f);
            cNor_.assign(3 * n, 0.f);
            cUv_.assign(2 * n, 0.f);
            cCol_.assign(4 * n, 1.f);
            std::vector<unsigned int> index;
            for (int k = 0; k < CurtainRows; ++k)
                for (int i = 0; i < nx; ++i) {
                    const unsigned a = k * (2 * nx) + 2 * i, b = a + 1, c = a + 2 * nx, d = c + 1;
                    index.insert(index.end(), {a, c, b, b, c, d});
                }
            auto g = threepp::BufferGeometry::create();
            g->setAttribute("position", threepp::FloatBufferAttribute::create(cPos_, 3));
            g->setAttribute("normal", threepp::FloatBufferAttribute::create(cNor_, 3));
            g->setAttribute("uv", threepp::FloatBufferAttribute::create(cUv_, 2));
            g->setAttribute("color", threepp::FloatBufferAttribute::create(cCol_, 4));
            for (const auto& name : {"position", "normal", "uv", "color"}) g->getAttribute(name)->setUsage(threepp::DrawUsage::Dynamic);
            g->setIndex(index);
            curtain = threepp::Mesh::create(g, material);
            curtain->frustumCulled = false;
            curtain->castShadow = false;
            curtain->receiveShadow = true;
            curtain->renderOrder = 3;
            updateCurtain();
        }

        std::vector<int> tail_;
        std::vector<float> positions_, normals_, velocity_, info_, uv_;
        std::shared_ptr<threepp::MeshStandardMaterial> material_;

        // the walls in the water: the bridge's piers (battered: wider at the foot), the wheel's pier, the mill
        static bool blocked(float x, float z) {
            using namespace site;
            for (float px : bridgePiers)
                if (std::abs(x - px) < pierHalfX + 0.12f && std::abs(z - bridgeZ) < pierHalfZ + 0.12f) return true;
            if (x > pierX0 - 0.03f && x < pierX1 + 0.03f && z > pierZ0 - 0.03f && z < pierZ1 + 0.03f) return true;
            if (inMill(x, z, 0.02f)) return true;
            return false;
        }
    };

    // ---- the flume's water ------------------------------------------------------------------------------------------
    // a steady sheet in the trough, running at `speed` toward the wheel, `depth` deep
    inline std::shared_ptr<threepp::Mesh> makeFlumeWater(const std::shared_ptr<threepp::Material>& material, float depth, float speed) {
        using namespace site;
        std::vector<float> pos, nor, vel, info, uv;
        std::vector<unsigned int> index;
        const int across = 4;
        const float hw = 0.5f * flumeInner - 0.012f;
        int rows = 0;
        for (float z = flumeStartZ; z <= flumeEndZ + 1e-3f; z += 0.25f, ++rows) {
            // over the last metre the water speeds up and thins toward the lip
            const float toLip = flumeEndZ - z;
            const float d = depth * (0.72f + 0.28f * smooth(0.f, 1.2f, toLip));
            for (int k = 0; k <= across; ++k) {
                const float x = flumeX - hw + 2.f * hw * static_cast<float>(k) / static_cast<float>(across);
                pos.insert(pos.end(), {x, flumeFloor(z) + d, z});
                nor.insert(nor.end(), {0.f, 1.f, 0.f});
                const float v = speed * depth / d;
                vel.insert(vel.end(), {0.f, v});
                info.insert(info.end(), {0.04f, d});
                uv.insert(uv.end(), {0.04f, 0.f});
            }
        }
        const int w = across + 1;
        for (int r = 0; r + 1 < rows; ++r)
            for (int k = 0; k < across; ++k) {
                const unsigned a = r * w + k, b = a + 1, c = a + w, d = c + 1;
                index.insert(index.end(), {a, c, b, b, c, d});
            }
        auto g = threepp::BufferGeometry::create();
        g->setAttribute("position", threepp::FloatBufferAttribute::create(pos, 3));
        g->setAttribute("normal", threepp::FloatBufferAttribute::create(nor, 3));
        g->setAttribute("flowVel", threepp::FloatBufferAttribute::create(vel, 2));
        g->setAttribute("waterInfo", threepp::FloatBufferAttribute::create(info, 2));
        g->setAttribute("uv", threepp::FloatBufferAttribute::create(uv, 2));
        g->setAttribute("uv2", threepp::FloatBufferAttribute::create(vel, 2));
        g->setIndex(index);
        auto mesh = threepp::Mesh::create(g, material);
        mesh->receiveShadow = true;
        return mesh;
    }

}// namespace mill

#endif
