// Procedural tree generator demo.
//
// Renders a single procedural tree (trunk + leaves) with a live ImGui panel
// that re-generates the tree when parameters change.  Four species presets
// (Oak, Pine, Birch, Willow) plus full manual control.

#include "threepp/extras/imgui/RendererSettings.hpp"
#include "threepp/extras/vegetation/TreeGenerator.hpp"
#include "threepp/extras/vegetation/TreeTextures.hpp"
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <cmath>
#include <random>

using namespace threepp;
using namespace threepp::vegetation;

int main() {

    Canvas canvas("Procedural Tree Generator", {{"vsync", true}, {"aa", 4}});
    auto renderer = createRenderer(canvas);
    renderer->setClearColor(Color(0.18f, 0.22f, 0.28f));
    renderer->toneMapping = ToneMapping::ACESFilmic;
    renderer->toneMappingExposure = 1.0f;
    renderer->shadowMap().enabled = true;
    renderer->shadowMap().type = ShadowMap::PFCSoft;

    Scene scene;

    // Image-based lighting from an outdoor sky HDR (drives ambient/IBL on
    // the MeshStandardMaterials) plus a shadow-casting key light = "sun".
    RGBELoader hdrLoader;
    if (auto hdr = hdrLoader.load(std::string(DATA_FOLDER) + "/textures/env/autumn_field_puresky_2k.hdr")) {
        scene.background = hdr;
        scene.environment = hdr;
    }

    auto ambient = AmbientLight::create(Color::white, 0.25f);
    scene.add(ambient);

    auto sun = DirectionalLight::create(Color(1.0f, 0.97f, 0.90f), 2.6f);
    sun->position.set(8.f, 16.f, 10.f);
    sun->castShadow = true;
    {
        auto* cam = sun->shadow->camera->as<OrthographicCamera>();
        cam->left = cam->bottom = -12.f;
        cam->right = cam->top = 12.f;
        cam->nearPlane = 1.f;
        cam->farPlane = 50.f;
        sun->shadow->mapSize.set(2048, 2048);
        sun->shadow->bias = -0.0004f;
    }
    scene.add(sun);

    // ── Procedural textures ──────────────────────────────────────────────
    TreeParams params;
    applyPreset(0, params);// start with Oak

    auto [barkAlbedo, barkNormal] = vegetation::makeBarkTextures(256, params.seed, params.barkColor);
    barkAlbedo->repeat.set(3.f, 0.5f);
    barkNormal->repeat.set(3.f, 0.5f);
    // Conifer fronds want the elongated needle cutout; broadleaf styles the
    // round leaf-cluster atlas.
    auto makeLeafTex = [](const TreeParams& p) {
        return p.leafStyle == LeafStyle::Frond
                ? vegetation::makeNeedleFrondTexture(256, p.seed, p.leafColor)
                : vegetation::makeLeafClusterTexture(256, p.seed, p.leafColor);
    };
    auto leafTex = makeLeafTex(params);

    // Ground plane (receives shadows).
    auto groundMat = MeshStandardMaterial::create(
            MeshStandardMaterial::Params{}
                    .color(Color(0.30f, 0.32f, 0.20f))
                    .roughness(1.0f)
                    .metalness(0.0f));
    auto ground = Mesh::create(PlaneGeometry::create(60.f, 60.f), groundMat);
    ground->rotation.x = -math::PI / 2.f;
    ground->position.y = 0.f;
    ground->receiveShadow = true;
    scene.add(ground);

    // Tree.
    TreeGenerator gen(params.seed);
    gen.buildSkeleton(params);

    auto barkMat = MeshStandardMaterial::create(
            MeshStandardMaterial::Params{}
                    .color(Color::white)
                    .roughness(0.92f)
                    .metalness(0.0f));
    barkMat->map = barkAlbedo;
    barkMat->normalMap = barkNormal;

    auto leafMat = MeshStandardMaterial::create(
            MeshStandardMaterial::Params{}
                    .color(Color::white)
                    .roughness(0.85f)
                    .metalness(0.0f));
    leafMat->map = leafTex;
    leafMat->alphaTest = 0.5f;
    leafMat->side = Side::Double;
    leafMat->vertexColors = true;// per-card tonal variation (top-lit gradient)
    // Foliage translucency: backlit canopy glow (Vulkan deferred). Live-editable
    // below — the material patch path propagates the value without a regen.
    leafMat->translucency = 0.45f;
    leafMat->translucencyColor = Color(0.55f, 0.85f, 0.30f);

    auto trunkMesh = Mesh::create(gen.makeTrunkGeometry(params), barkMat);
    auto leafMesh = Mesh::create(gen.makeLeafGeometry(params), leafMat);
    trunkMesh->castShadow = true;
    trunkMesh->receiveShadow = true;
    leafMesh->castShadow = true;
    leafMesh->receiveShadow = true;
    scene.add(trunkMesh);
    scene.add(leafMesh);

    // Camera.
    PerspectiveCamera camera(50.f, canvas.aspect(), 0.1f, 200.f);
    camera.position.set(8.f, 6.f, 8.f);
    OrbitControls controls{camera, canvas};
    controls.target.set(0.f, 4.f, 0.f);
    controls.update();

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer->setSize(size);
    });

    // ── State ────────────────────────────────────────────────────────────
    int preset = 0;
    bool regenRequested = false;
    int crownShapeIdx = static_cast<int>(params.crownShape);
    int leafStyleIdx = static_cast<int>(params.leafStyle);

    auto regenerate = [&] {
        gen.reseed(params.seed);
        gen.buildSkeleton(params);
        trunkMesh->setGeometry(gen.makeTrunkGeometry(params));
        leafMesh->setGeometry(gen.makeLeafGeometry(params));

        // Albedo/colour lives in the procedural textures — rebuild them.
        auto bark = vegetation::makeBarkTextures(256, params.seed, params.barkColor);
        bark.first->repeat.set(3.f, 0.5f);
        bark.second->repeat.set(3.f, 0.5f);
        barkMat->map = bark.first;
        barkMat->normalMap = bark.second;
        barkMat->needsUpdate();

        leafMat->map = makeLeafTex(params);
        leafMat->needsUpdate();
    };

    auto markCustom = [&](bool changed) {
        if (changed) {
            preset = 4;
            regenRequested = true;
        }
    };

    // ── ImGui ────────────────────────────────────────────────────────────
    // Generic renderer settings (tone map, shadows, ...) come from the shared
    // panel; the tree parameters are the app-specific widgets below.
    RendererSettingsUi ui(canvas, *renderer, [&] {
        ImGui::Text("nodes: %d", gen.nodeCount());
        ImGui::Separator();

        // Preset.
        if (ImGui::Combo("Preset", &preset, "Oak\0Pine\0Birch\0Willow\0Custom\0")) {
            if (preset < 4) {
                applyPreset(preset, params);
                crownShapeIdx = static_cast<int>(params.crownShape);
                leafStyleIdx = static_cast<int>(params.leafStyle);
                regenRequested = true;
            }
        }

        // Seed.
        {
            int seedI = static_cast<int>(params.seed);
            if (ImGui::InputInt("Seed", &seedI)) {
                params.seed = static_cast<unsigned int>(std::max(seedI, 0));
                regenRequested = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Randomize")) {
                params.seed = std::random_device{}();
                regenRequested = true;
            }
        }

        ImGui::SeparatorText("Trunk");
        markCustom(ImGui::SliderFloat("Trunk height", &params.trunkHeight, 0.5f, 12.f, "%.1f"));
        markCustom(ImGui::SliderFloat("Trunk radius", &params.trunkRadius, 0.02f, 0.5f, "%.3f"));

        ImGui::SeparatorText("Crown");
        if (ImGui::Combo("Shape", &crownShapeIdx, "Sphere\0Ellipsoid\0Cone\0Hemisphere\0Cylinder\0")) {
            params.crownShape = static_cast<CrownShape>(crownShapeIdx);
            preset = 4;
            regenRequested = true;
        }
        markCustom(ImGui::SliderFloat("Radius X", &params.crownRadiusX, 0.5f, 10.f, "%.1f"));
        markCustom(ImGui::SliderFloat("Radius Z", &params.crownRadiusZ, 0.5f, 10.f, "%.1f"));
        markCustom(ImGui::SliderFloat("Height", &params.crownHeight, 1.f, 15.f, "%.1f"));

        ImGui::SeparatorText("Colonisation");
        markCustom(ImGui::SliderInt("Attractors", &params.attractorCount, 100, 3000));
        markCustom(ImGui::SliderFloat("Influence dist", &params.influenceDistance, 1.f, 10.f, "%.1f"));
        markCustom(ImGui::SliderFloat("Kill dist", &params.killDistance, 0.2f, 3.f, "%.2f"));
        markCustom(ImGui::SliderFloat("Segment length", &params.segmentLength, 0.1f, 1.5f, "%.2f"));
        markCustom(ImGui::SliderInt("Max iterations", &params.maxIterations, 50, 500));
        markCustom(ImGui::SliderFloat("Randomness", &params.randomness, 0.f, 0.3f, "%.3f"));
        markCustom(ImGui::SliderFloat("Tropism", &params.tropism, -0.2f, 0.1f, "%.3f"));

        ImGui::SeparatorText("Branch geometry");
        markCustom(ImGui::SliderFloat("Radius exp", &params.radiusExponent, 1.5f, 4.f, "%.1f"));
        markCustom(ImGui::SliderFloat("Min branch r", &params.minBranchRadius, 0.001f, 0.02f, "%.4f"));
        markCustom(ImGui::SliderInt("Radial segs", &params.radialSegments, 3, 12));

        ImGui::SeparatorText("Trunk & bark variation");
        markCustom(ImGui::SliderFloat("Lean", &params.trunkLean, 0.f, 0.25f, "%.3f"));
        markCustom(ImGui::SliderFloat("Bend", &params.trunkBend, 0.f, 3.f, "%.2f"));
        markCustom(ImGui::SliderFloat("Twist", &params.trunkTwist, -1.5f, 1.5f, "%.2f"));
        markCustom(ImGui::SliderFloat("Bark bump", &params.barkBumpAmp, 0.f, 0.3f, "%.3f"));
        markCustom(ImGui::SliderInt("Bark lobes", &params.barkBumpLobes, 2, 12));
        markCustom(ImGui::SliderFloat("Root flare", &params.rootFlareAsym, 0.f, 1.f, "%.2f"));

        ImGui::SeparatorText("Conifer (whorl mode)");
        {
            int modeIdx = static_cast<int>(params.branchingMode);
            if (ImGui::Combo("Branching", &modeIdx, "Colonise\0Whorl\0")) {
                params.branchingMode = static_cast<BranchingMode>(modeIdx);
                preset = 4;
                regenRequested = true;
            }
        }
        markCustom(ImGui::SliderFloat("Whorl spacing", &params.whorlSpacing, 0.3f, 1.5f, "%.2f"));
        markCustom(ImGui::SliderInt("Branches/whorl", &params.branchesPerWhorl, 2, 10));
        markCustom(ImGui::SliderFloat("Whorl jitter", &params.whorlJitter, 0.f, 1.f, "%.2f"));
        markCustom(ImGui::SliderFloat("Branch droop", &params.branchDroop, 0.f, 1.f, "%.2f"));
        markCustom(ImGui::SliderFloat("Tip upturn", &params.branchTipUpturn, 0.f, 1.f, "%.2f"));
        markCustom(ImGui::SliderFloat("Crown profile", &params.crownProfileExponent, 0.5f, 3.f, "%.2f"));
        markCustom(ImGui::SliderFloat("Side twigs", &params.sideTwigDensity, 0.f, 1.f, "%.2f"));

        ImGui::SeparatorText("Leaves");
        if (ImGui::Combo("Leaf style", &leafStyleIdx, "Quad\0Cluster\0CrossQuad\0Blob\0Frond\0")) {
            params.leafStyle = static_cast<LeafStyle>(leafStyleIdx);
            preset = 4;
            regenRequested = true;
        }
        markCustom(ImGui::SliderFloat("Leaf size", &params.leafSize, 0.05f, 1.0f, "%.2f"));
        markCustom(ImGui::SliderFloat("Leaf density", &params.leafDensity, 0.f, 1.f, "%.2f"));
        markCustom(ImGui::SliderFloat("Clumping", &params.leafClumping, 0.f, 0.9f, "%.2f"));
        markCustom(ImGui::SliderInt("Per cluster", &params.leavesPerCluster, 1, 10));
        markCustom(ImGui::SliderFloat("Leaf spread", &params.leafSpread, 0.f, 1.f, "%.2f"));

        // Foliage translucency (live — no regen; exercises the material patch path).
        ImGui::SliderFloat("Translucency", &leafMat->translucency, 0.f, 1.f, "%.2f");
        if (ImGui::ColorEdit3("Translucency tint", &leafMat->translucencyColor.r))
            leafMat->needsUpdate();

        ImGui::SeparatorText("Colors");
        if (ImGui::ColorEdit3("Bark", params.barkColor.data())) { preset = 4; regenRequested = true; }
        if (ImGui::ColorEdit3("Leaf", params.leafColor.data())) { preset = 4; regenRequested = true; }

        if (ImGui::Button("Generate", ImVec2(-1, 0))) regenRequested = true;
    }, "Tree Configurator");

    // ── Render loop ──────────────────────────────────────────────────────
    canvas.animate([&] {
        controls.update();

        if (regenRequested && !ImGui::IsAnyItemActive()) {
            regenerate();
            regenRequested = false;
        }

        renderer->render(scene, camera);
        ui.render();
    });
}
