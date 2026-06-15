// Procedural forest demo.
//
// Combines the two procedural generators:
//   - terrain::TerrainGenerator  → a rolling heightfield with a slope/altitude
//                                   splat albedo map.
//   - vegetation::TreeGenerator  → a handful of prototype trees (trunk tubes +
//                                   alpha-cutout leaf cards) that are scattered
//                                   across the terrain with InstancedMesh, so a
//                                   few hundred trees cost only a few draw calls.
//
// Trees are placed on a jittered grid, dropped onto the terrain surface, and
// rejected on steep slopes.  Lit by an outdoor-sky HDR (IBL) + a shadow-casting
// sun, with distance fog for depth.

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/extras/terrain/TerrainGenerator.hpp"
#include "threepp/extras/vegetation/TreeGenerator.hpp"
#include "threepp/extras/vegetation/TreeTextures.hpp"
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/textures/DataTexture.hpp"
#include "threepp/threepp.hpp"

#ifdef THREEPP_WITH_VULKAN
#include "threepp/renderers/VulkanRenderer.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

using namespace threepp;

namespace {

    // One prototype tree: shared geometry + materials, instanced many times.
    struct TreeVariant {
        std::shared_ptr<BufferGeometry> trunkGeo;
        std::shared_ptr<BufferGeometry> leafGeo;
        std::shared_ptr<MeshStandardMaterial> barkMat;
        std::shared_ptr<MeshStandardMaterial> leafMat;
    };

    // ── Grass ────────────────────────────────────────────────────────────
    // A single tapered blade, instanced thousands of times. The sway is a
    // whole-blade tilt applied by rewriting each instance matrix on the CPU
    // every frame — this works on BOTH the GL and Vulkan backends (Vulkan has
    // no ShaderMaterial path, but it consumes InstancedMesh matrices directly).
    // A baked vertex-colour gradient + a standard lit material give the blades
    // shading and scene fog for free.
    std::shared_ptr<BufferGeometry> makeGrassBlade() {
        constexpr int seg = 4;
        constexpr float wBase = 0.05f;// half-width at the base
        // Kept deliberately dark: strong sun + sky IBL otherwise pushes the
        // green past 1.0 and ACES desaturates the tips toward white.
        const Vector3 bottom{0.06f, 0.13f, 0.04f};
        const Vector3 top{0.20f, 0.34f, 0.11f};
        std::vector<float> pos, nrm, uv, col;
        std::vector<unsigned int> idx;
        for (int i = 0; i <= seg; ++i) {
            const float t = static_cast<float>(i) / static_cast<float>(seg);
            const float y = t;                 // unit height (scaled per instance)
            const float w = wBase * (1.f - t); // taper to a point at the tip
            const float r = bottom.x + (top.x - bottom.x) * t;
            const float g = bottom.y + (top.y - bottom.y) * t;
            const float b = bottom.z + (top.z - bottom.z) * t;
            for (int s = 0; s < 2; ++s) {
                const float x = (s == 0) ? -w : w;
                pos.push_back(x);
                pos.push_back(y);
                pos.push_back(0.f);
                // Up-biased normal: catches sky/sun light regardless of facing.
                nrm.push_back(0.f);
                nrm.push_back(0.85f);
                nrm.push_back(0.53f);
                uv.push_back(s == 0 ? 0.f : 1.f);
                uv.push_back(t);
                col.push_back(r);
                col.push_back(g);
                col.push_back(b);
            }
        }
        for (int i = 0; i < seg; ++i) {
            const auto a = static_cast<unsigned int>(i * 2);
            const unsigned int b = a + 1, c = a + 2, d = a + 3;
            idx.push_back(a); idx.push_back(b); idx.push_back(c);
            idx.push_back(b); idx.push_back(d); idx.push_back(c);
        }
        auto geo = BufferGeometry::create();
        geo->setIndex(idx);
        geo->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        geo->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
        geo->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
        geo->setAttribute("color", FloatBufferAttribute::create(col, 3));
        return geo;
    }

    // Per-blade static placement data; used by the CPU (Vulkan) fallback,
    // which recomputes the tilt matrix each frame.
    struct Blade {
        Vector3 pos;
        Vector3 scale;
        Quaternion yaw;// base orientation about +Y
        float phase;   // wind phase offset
    };

    // ── Preferred path: GPU vertex-shader wind (GL + WGPU via GLSL compat) ─
    // Cheaper (no per-frame CPU work) and nicer (per-vertex curve, base
    // planted) than the CPU tilt. ShaderMaterial supplies the #version,
    // modelViewMatrix / projectionMatrix / instanceMatrix / position / uv.
    const char* grassVertexShader() {
        return R"(
            uniform float time;
            uniform float windStrength;
            uniform vec2  windDir;
            varying float vHeight;
            varying float vFog;
            void main() {
                vHeight = uv.y;
                vec3 p = position;
            #ifdef USE_INSTANCING
                vec3 instPos = vec3(instanceMatrix[3][0], instanceMatrix[3][1], instanceMatrix[3][2]);
            #else
                vec3 instPos = vec3(0.0);
            #endif
                float phase = time * 1.6 + instPos.x * 0.25 + instPos.z * 0.25;
                float gust  = sin(phase) * 0.6 + sin(phase * 2.3 + 1.7) * 0.25;
                float bend  = gust * windStrength * vHeight * vHeight;// base planted, tip sways
                p.x += windDir.x * bend;
                p.z += windDir.y * bend;
            #ifdef USE_INSTANCING
                vec4 mv = modelViewMatrix * instanceMatrix * vec4(p, 1.0);
            #else
                vec4 mv = modelViewMatrix * vec4(p, 1.0);
            #endif
                vFog = -mv.z;
                gl_Position = projectionMatrix * mv;
            }
        )";
    }

    const char* grassFragmentShader() {
        // Self-contained shading (no IBL) keeps the blades from blowing out to
        // white the way a full standard material under the bright sky does.
        return R"(
            uniform vec3  topColor;
            uniform vec3  bottomColor;
            uniform vec3  sunDir;
            uniform vec3  sunColor;
            uniform vec3  ambient;
            uniform vec3  fogColor;
            uniform float fogNear;
            uniform float fogFar;
            varying float vHeight;
            varying float vFog;
            void main() {
                vec3 base = mix(bottomColor, topColor, vHeight);
                // Thin blades: a soft constant wrap + ambient, no harsh N·L term.
                vec3 lit = base * (ambient + sunColor * 0.7);
                float f = clamp((vFog - fogNear) / (fogFar - fogNear), 0.0, 1.0);
                gl_FragColor = vec4(mix(lit, fogColor, f), 1.0);
            }
        )";
    }

    TreeVariant makeVariant(int preset, unsigned int seed) {
        vegetation::TreeParams tp;
        vegetation::applyPreset(preset, tp);
        tp.seed = seed;

        if (preset == 2) {
            // Birch: the pure-white preset bark reads as a bare pole at forest
            // distance — mute it to a soft grey-birch and fill the crown more.
            tp.barkColor = {0.70f, 0.69f, 0.66f};
            tp.leafDensity = 0.97f;
            tp.leafClumping = 0.35f;
        }

        vegetation::TreeGenerator gen(seed);
        gen.buildSkeleton(tp);

        TreeVariant v;
        v.trunkGeo = gen.makeTrunkGeometry(tp);
        v.leafGeo = gen.makeLeafGeometry(tp);

        auto bark = vegetation::makeBarkTextures(256, seed, tp.barkColor);
        bark.first->repeat.set(3.f, 0.5f);
        bark.second->repeat.set(3.f, 0.5f);
        v.barkMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color::white).roughness(0.92f).metalness(0.f));
        v.barkMat->map = bark.first;
        v.barkMat->normalMap = bark.second;

        v.leafMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color::white).roughness(0.85f).metalness(0.f));
        v.leafMat->map = vegetation::makeLeafClusterTexture(256, seed, tp.leafColor);
        v.leafMat->alphaTest = 0.5f;
        v.leafMat->side = Side::Double;
        v.leafMat->vertexColors = true;
        return v;
    }

}// namespace

int main() {

    Canvas canvas("Procedural Forest", {{"vsync", true}, {"aa", 4}});
    auto renderer = createRenderer(canvas);

    // GPU vertex-shader grass works on GL + WGPU (GLSL compat); only the
    // Vulkan backend lacks a ShaderMaterial path, so it uses the CPU fallback.
    bool shaderGrass = true;
#ifdef THREEPP_WITH_VULKAN
    if (dynamic_cast<VulkanRenderer*>(renderer.get())) shaderGrass = false;
#endif

    renderer->setClearColor(Color(0.62f, 0.72f, 0.84f));
    renderer->toneMapping = ToneMapping::ACESFilmic;
    renderer->toneMappingExposure = 1.0f;
    renderer->shadowMap().enabled = true;
    renderer->shadowMap().type = ShadowMap::PFCSoft;

    Scene scene;

    // ── Lighting: sky HDR (IBL) + shadow-casting sun ─────────────────────
    RGBELoader hdrLoader;
    if (auto hdr = hdrLoader.load(std::string(DATA_FOLDER) + "/textures/env/autumn_field_puresky_2k.hdr")) {
        scene.background = hdr;
        scene.environment = hdr;
    }

    auto ambient = AmbientLight::create(Color::white, 0.25f);
    scene.add(ambient);

    auto sun = DirectionalLight::create(Color(1.0f, 0.97f, 0.90f), 2.8f);
    sun->position.set(60.f, 120.f, 80.f);
    sun->castShadow = true;
    {
        auto* cam = sun->shadow->camera->as<OrthographicCamera>();
        cam->left = cam->bottom = -90.f;
        cam->right = cam->top = 90.f;
        cam->nearPlane = 1.f;
        cam->farPlane = 400.f;
        sun->shadow->mapSize.set(4096, 4096);
        sun->shadow->bias = -0.0005f;
    }
    scene.add(sun);

    // ── Terrain ──────────────────────────────────────────────────────────
    terrain::TerrainParams terr;
    terr.seed = 20260615u;
    terr.worldSize = 260.f;
    terr.resolution = 220;
    terr.noiseType = terrain::NoiseType::fBm;// gentle rolling hills (no erosion → heightAt matches mesh)
    terr.featureScale = 110.f;
    terr.octaves = 6;
    terr.amplitude = 16.f;
    terr.warp = 0.3f;
    terr.heightExponent = 1.0f;
    terr.erosion = terrain::ErosionType::None;
    // Grassy meadow palette.
    terr.grassColor = {0.27f, 0.34f, 0.17f};
    terr.screeColor = {0.45f, 0.42f, 0.34f};
    terr.rockColor = {0.40f, 0.37f, 0.33f};
    terr.snowLine = 2.0f;// effectively no snow
    terr.slopeGrassMax = 0.55f;

    terrain::TerrainGenerator terrainGen(terr.seed);
    auto terrainMat = MeshStandardMaterial::create(
            MeshStandardMaterial::Params{}.color(Color::white).roughness(0.97f).metalness(0.f));
    auto terrainMesh = Mesh::create(terrainGen.createGeometry(terr, false), terrainMat);
    {
        auto tex = DataTexture::create(ImageData{terrainGen.bakeSplatColors(terr)},
                                       static_cast<unsigned int>(terrainGen.dim()),
                                       static_cast<unsigned int>(terrainGen.dim()));
        tex->colorSpace = ColorSpace::sRGB;
        tex->magFilter = Filter::Linear;
        tex->minFilter = Filter::Linear;
        terrainMat->map = tex;
    }
    terrainMesh->receiveShadow = true;
    scene.add(terrainMesh);

    // ── Tree prototypes ──────────────────────────────────────────────────
    std::vector<TreeVariant> variants;
    variants.push_back(makeVariant(0, 101u));// Oak
    variants.push_back(makeVariant(0, 202u));// Oak (alt)
    variants.push_back(makeVariant(1, 303u));// Pine
    variants.push_back(makeVariant(1, 505u));// Pine (alt)
    variants.push_back(makeVariant(2, 404u));// Birch (accent — 1 in 5)

    // ── Scatter trees onto the terrain (jittered grid + slope rejection) ──
    int treeCount = 0;
    std::vector<std::shared_ptr<InstancedMesh>> forest;

    auto buildForest = [&](unsigned int scatterSeed, float spacing, float fillProb) {
        for (auto& m : forest) scene.remove(*m);
        forest.clear();
        treeCount = 0;

        std::mt19937 rng(scatterSeed);
        std::uniform_real_distribution<float> u01(0.f, 1.f);

        const float half = terr.worldSize * 0.5f * 0.92f;
        const int cells = std::max(1, static_cast<int>((2.f * half) / spacing));

        // Collect per-variant instance transforms.
        std::vector<std::vector<Matrix4>> xforms(variants.size());
        Quaternion q;
        const Vector3 up{0.f, 1.f, 0.f};

        for (int cz = 0; cz < cells; ++cz) {
            for (int cx = 0; cx < cells; ++cx) {
                if (u01(rng) > fillProb) continue;
                const float jx = (u01(rng) - 0.5f) * spacing * 0.8f;
                const float jz = (u01(rng) - 0.5f) * spacing * 0.8f;
                const float x = -half + (static_cast<float>(cx) + 0.5f) * (2.f * half / static_cast<float>(cells)) + jx;
                const float z = -half + (static_cast<float>(cz) + 0.5f) * (2.f * half / static_cast<float>(cells)) + jz;

                // Surface height + slope (central differences on procedural height).
                const float e = 1.0f;
                const float h = terrainGen.heightAt(x, z, terr);
                const float hx = terrainGen.heightAt(x + e, z, terr) - terrainGen.heightAt(x - e, z, terr);
                const float hz = terrainGen.heightAt(x, z + e, terr) - terrainGen.heightAt(x, z - e, terr);
                const float ny = (2.f * e) / std::sqrt(hx * hx + hz * hz + (2.f * e) * (2.f * e));
                if (ny < 0.86f) continue;// too steep — bare slope

                const size_t vi = static_cast<size_t>(u01(rng) * static_cast<float>(variants.size())) % variants.size();
                const float s = 1.1f + u01(rng) * 0.9f;// scale variety
                q.setFromAxisAngle(up, u01(rng) * 6.28318530718f);
                Matrix4 m;
                m.compose(Vector3(x, h - 0.2f, z), q, Vector3(s, s, s));
                xforms[vi].push_back(m);
            }
        }

        // Build one trunk + one leaf InstancedMesh per variant.
        for (size_t vi = 0; vi < variants.size(); ++vi) {
            const auto& xf = xforms[vi];
            if (xf.empty()) continue;
            treeCount += static_cast<int>(xf.size());

            auto trunks = InstancedMesh::create(variants[vi].trunkGeo, variants[vi].barkMat, xf.size());
            auto leaves = InstancedMesh::create(variants[vi].leafGeo, variants[vi].leafMat, xf.size());
            trunks->castShadow = true;
            trunks->receiveShadow = true;
            leaves->castShadow = true;
            for (size_t i = 0; i < xf.size(); ++i) {
                trunks->setMatrixAt(i, xf[i]);
                leaves->setMatrixAt(i, xf[i]);
            }
            trunks->instanceMatrix()->needsUpdate();
            leaves->instanceMatrix()->needsUpdate();
            scene.add(trunks);
            scene.add(leaves);
            forest.push_back(trunks);
            forest.push_back(leaves);
        }
    };

    float spacing = 11.f;
    float fillProb = 0.8f;
    unsigned int scatterSeed = 1u;
    buildForest(scatterSeed, spacing, fillProb);

    // ── Distance fog (shared by the scene and the grass shader) ──────────
    const Color fogColor(0.66f, 0.75f, 0.86f);
    const float fogNear = terr.worldSize * 0.35f;
    const float fogFar = terr.worldSize * 1.05f;
    scene.fog = Fog(fogColor, fogNear, fogFar);

    // ── Swaying grass (instanced) ────────────────────────────────────────
    const int bladeCount = shaderGrass ? 90000 : 45000;// GPU path is cheap → denser
    const float grassRadius = 70.f;                    // camera-centred patch
    const Vector3 windAxis = Vector3(0.6f, 0.f, -0.8f).normalize();// CPU-path tilt axis ⟂ wind
    const Vector2 windDir2(0.8f, 0.6f);

    // Material: ShaderMaterial (GPU wind) when supported, else a standard lit
    // material driven by the CPU tilt below.
    std::shared_ptr<ShaderMaterial> grassShaderMat;
    std::shared_ptr<Material> grassMat;
    if (shaderGrass) {
        Vector3 sunDirN(60.f, 120.f, 80.f);
        sunDirN.normalize();
        grassShaderMat = ShaderMaterial::create();
        grassShaderMat->vertexShader = grassVertexShader();
        grassShaderMat->fragmentShader = grassFragmentShader();
        grassShaderMat->side = Side::Double;
        grassShaderMat->uniforms["time"].setValue(0.f);
        grassShaderMat->uniforms["windStrength"].setValue(0.18f);
        grassShaderMat->uniforms["windDir"].setValue(windDir2);
        grassShaderMat->uniforms["topColor"].setValue(Vector3(0.30f, 0.42f, 0.14f));
        grassShaderMat->uniforms["bottomColor"].setValue(Vector3(0.08f, 0.16f, 0.05f));
        grassShaderMat->uniforms["sunColor"].setValue(Vector3(0.55f, 0.55f, 0.50f));
        grassShaderMat->uniforms["ambient"].setValue(Vector3(0.30f, 0.34f, 0.30f));
        grassShaderMat->uniforms["fogColor"].setValue(Vector3(fogColor.r, fogColor.g, fogColor.b));
        grassShaderMat->uniforms["fogNear"].setValue(fogNear);
        grassShaderMat->uniforms["fogFar"].setValue(fogFar);
        grassMat = grassShaderMat;
    } else {
        auto std = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color::white).roughness(0.97f).metalness(0.f));
        std->vertexColors = true;
        std->side = Side::Double;
        std->envMapIntensity = 0.45f;
        grassMat = std;
    }

    auto grass = InstancedMesh::create(makeGrassBlade(), grassMat, static_cast<size_t>(bladeCount));
    std::vector<Blade> blades(static_cast<size_t>(bladeCount));
    {
        std::mt19937 grng(7u);
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        const Vector3 up{0.f, 1.f, 0.f};
        Matrix4 m;
        for (int i = 0; i < bladeCount; ++i) {
            const float ang = u01(grng) * 6.28318530718f;
            const float rr = std::sqrt(u01(grng)) * grassRadius;// uniform over disk
            const float x = std::cos(ang) * rr;
            const float z = std::sin(ang) * rr;
            const float h = terrainGen.heightAt(x, z, terr);
            const float s = 0.5f + u01(grng) * 0.5f;  // width scale
            const float hgt = 0.3f + u01(grng) * 0.4f;// height 0.3–0.7 world units

            Blade& bl = blades[static_cast<size_t>(i)];
            bl.pos.set(x, h - 0.05f, z);
            bl.scale.set(s, hgt, s);
            bl.yaw.setFromAxisAngle(up, u01(grng) * 6.28318530718f);
            bl.phase = u01(grng) * 6.28318530718f;
            m.compose(bl.pos, bl.yaw, bl.scale);
            grass->setMatrixAt(static_cast<size_t>(i), m);
        }
        grass->instanceMatrix()->needsUpdate();// static for the shader path
    }
    scene.add(grass);

    // ── Camera ───────────────────────────────────────────────────────────
    PerspectiveCamera camera(55.f, canvas.aspect(), 0.1f, 1000.f);
    camera.position.set(70.f, 38.f, 70.f);
    OrbitControls controls{camera, canvas};
    controls.target.set(0.f, 6.f, 0.f);
    controls.update();

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer->setSize(size);
    });

    // ── UI ───────────────────────────────────────────────────────────────
    float fps = 0.f, fpsAccum = 0.f;
    int fpsFrames = 0;
    bool regen = false;
    float windStrength = 0.18f;

    ImguiFunctionalContext ui(canvas, *renderer, [&] {
        ImGui::SetNextWindowPos({10, 10}, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize({300, 0}, ImGuiCond_FirstUseEver);
        ImGui::Begin("Forest");
        ImGui::Text("FPS: %.1f   trees: %d", fps, treeCount);
        ImGui::Separator();
        ImGui::SliderFloat("Spacing", &spacing, 6.f, 24.f, "%.1f");
        ImGui::SliderFloat("Fill", &fillProb, 0.1f, 1.0f, "%.2f");
        if (ImGui::Button("Re-scatter", ImVec2(-1, 0))) {
            ++scatterSeed;
            regen = true;
        }
        ImGui::SeparatorText("Wind");
        ImGui::SliderFloat("Strength", &windStrength, 0.f, 0.5f, "%.2f");
        ImGui::End();
    });

    IOCapture ioCapture;
    ioCapture.preventMouseEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    ioCapture.preventScrollEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    ioCapture.preventKeyboardEvent = [] { return ImGui::GetIO().WantCaptureKeyboard; };
    canvas.setIOCapture(&ioCapture);

    Clock clock;
    float tElapsed = 0.f;
    Matrix4 m;
    Quaternion qLean, qOut;
    canvas.animate([&] {
        const float dt = clock.getDelta();
        tElapsed += dt;
        controls.update();

        // Wind sway.
        if (shaderGrass) {
            // GPU path: just advance the shader clock (no per-frame CPU work).
            grassShaderMat->uniforms["time"].setValue(tElapsed);
            grassShaderMat->uniforms["windStrength"].setValue(windStrength);
        } else {
            // CPU fallback (Vulkan): rewrite each instance matrix as a whole-
            // blade tilt toward the wind.
            for (size_t i = 0; i < blades.size(); ++i) {
                const Blade& bl = blades[i];
                const float gust = std::sin(tElapsed * 1.6f + bl.phase) * 0.6f +
                                   std::sin(tElapsed * 3.7f + bl.phase * 2.3f) * 0.25f;
                qLean.setFromAxisAngle(windAxis, gust * windStrength);
                qOut.multiplyQuaternions(qLean, bl.yaw);
                m.compose(bl.pos, qOut, bl.scale);
                grass->setMatrixAt(i, m);
            }
            grass->instanceMatrix()->needsUpdate();
        }

        fpsAccum += dt;
        fpsFrames++;
        if (fpsAccum >= 0.5f) {
            fps = static_cast<float>(fpsFrames) / fpsAccum;
            fpsAccum = 0.f;
            fpsFrames = 0;
        }

        if (regen && !ImGui::IsAnyItemActive()) {
            buildForest(scatterSeed, spacing, fillProb);
            regen = false;
        }

        renderer->render(scene, camera);
        ui.render();
    });
}
