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
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/textures/DataTexture.hpp"
#include "threepp/threepp.hpp"

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

    // ── Distance fog for depth ───────────────────────────────────────────
    scene.fog = Fog(Color(0.66f, 0.75f, 0.86f), terr.worldSize * 0.35f, terr.worldSize * 1.05f);

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
        ImGui::End();
    });

    IOCapture ioCapture;
    ioCapture.preventMouseEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    ioCapture.preventScrollEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    ioCapture.preventKeyboardEvent = [] { return ImGui::GetIO().WantCaptureKeyboard; };
    canvas.setIOCapture(&ioCapture);

    Clock clock;
    canvas.animate([&] {
        const float dt = clock.getDelta();
        controls.update();

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
