// GTAOPass: ground-truth ambient occlusion. Creases, corners and the spots
// where things touch the floor go darker, the way they do under an overcast
// sky - light from the environment cannot get into them.
//
// The scene is lit by an environment map only (plus a weak key light), which
// is where AO matters most: with no direct shadows, nothing else tells you the
// boxes are standing on the floor.
//
//   G        AO on / off
//   M        cycle the output: Default, Diffuse, AO, Denoise, Depth, Normal
//
// Every knob of the pass and of its denoiser is in the panel. The ones worth
// trying first, with the output on Denoise: radius (how large a crease
// darkens) and thickness (how far behind a silhouette still gets darkened by
// it - raise it and every outline grows a halo, lower it and only the contacts
// are left). The AO output is the same image before the denoiser.

#include "threepp/extras/environments/RoomEnvironment.hpp"
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/geometries/TorusKnotGeometry.hpp"
#include "threepp/input/KeyListener.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/postprocessing/postprocessing.hpp"
#include "threepp/renderers/gl/PMREMGenerator.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

using namespace threepp;

namespace {

    std::shared_ptr<Mesh> solid(const std::shared_ptr<BufferGeometry>& geometry, const Color& color,
                                const Vector3& position, float roughness = 0.7f) {

        auto material = MeshStandardMaterial::create();
        material->color = color;
        material->roughness = roughness;

        auto mesh = Mesh::create(geometry, material);
        mesh->position.copy(position);

        return mesh;
    }

    // Moves the object straight up or down until its lowest point is at
    // `height`: resting on a surface, neither sunk into it nor hovering.
    void restOn(Object3D& object, float height) {

        Box3 box;
        box.setFromObject(object, true);
        object.position.y += height - box.min().y;
    }

    const std::array<std::pair<GTAOPass::Output, const char*>, 6> outputs{{
            {GTAOPass::Output::Default, "Default"},
            {GTAOPass::Output::Diffuse, "Diffuse"},
            {GTAOPass::Output::AO, "AO"},
            {GTAOPass::Output::Denoise, "Denoise"},
            {GTAOPass::Output::Depth, "Depth"},
            {GTAOPass::Output::Normal, "Normal"},
    }};

}// namespace


int main(int argc, char** argv) {

    // Headless capture (dev):
    //   gtao --shot <name.png> [--mode default|diffuse|ao|denoise|depth|normal]
    //        [--off] [--ortho] [--intensity X] [--size WxH] [--ui]
    std::string shotPath;
    int modeIndex = 0;
    bool startOff = false, ortho = false, shotWithUi = false;
    float intensity = 1.f;
    int width = 1280, height = 720;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shot" && i + 1 < argc) shotPath = argv[++i];
        else if (a == "--off") startOff = true;
        else if (a == "--ortho") ortho = true;
        else if (a == "--ui") shotWithUi = true;
        else if (a == "--intensity" && i + 1 < argc) intensity = std::stof(argv[++i]);
        else if (a == "--size" && i + 1 < argc) std::sscanf(argv[++i], "%dx%d", &width, &height);
        else if (a == "--mode" && i + 1 < argc) {
            std::string m = argv[++i];
            for (size_t k = 0; k < outputs.size(); k++) {
                std::string name = outputs[k].second;
                std::transform(name.begin(), name.end(), name.begin(), ::tolower);
                if (name == m) modeIndex = static_cast<int>(k);
            }
        }
    }
    const bool capture = !shotPath.empty();

    Canvas canvas(Canvas::Parameters()
                          .title("GTAO")
                          .size(width, height)
                          .antialiasing(0)
                          .headless(capture));
    GLRenderer renderer(canvas);
    renderer.toneMapping = ToneMapping::ACESFilmic;

    auto scene = Scene::create();
    scene->background = Color(0xbfe3dd);
    {
        PMREMGenerator pmrem(renderer);
        RoomEnvironment room;
        scene->environment = pmrem.fromScene(room, 0.04f);
    }

    // A far plane just past the room: the Depth output spreads near..far over
    // black..white, and a distant far plane leaves the whole room near white.
    std::shared_ptr<Camera> camera;
    if (ortho) {
        const float h = 4.f, w = h * canvas.aspect();
        camera = OrthographicCamera::create(-w, w, h, -h, 0.1f, 20.f);
    } else {
        camera = PerspectiveCamera::create(40, canvas.aspect(), 0.1f, 20.f);
    }
    camera->position.set(3.4f, 2.7f, 5.0f);

    OrbitControls controls{*camera, canvas};
    controls.target.set(-0.8f, 0.6f, -0.8f);
    controls.maxDistance = 11.f;// keeps the room inside the far plane
    controls.update();

    auto key = DirectionalLight::create(0xffffff, 0.6f);
    key->position.set(3, 6, 4);
    scene->add(key);

    const Color wallColor(0xd8d4cc);
    auto floor = solid(BoxGeometry::create(10, 0.2f, 10), wallColor, {0, -0.1f, 0});
    scene->add(floor);
    scene->add(solid(BoxGeometry::create(0.2f, 4, 10), wallColor, {-3.1f, 2, 0}));// side wall
    scene->add(solid(BoxGeometry::create(10, 4, 0.2f), wallColor, {0, 2, -3.1f}));// back wall

    // Boxes stacked into the corner.
    scene->add(solid(BoxGeometry::create(1.2f, 1.2f, 1.2f), Color(0xc05040), {-2.4f, 0.6f, -2.4f}));
    scene->add(solid(BoxGeometry::create(0.8f, 0.8f, 0.8f), Color(0x4070c0), {-2.6f, 1.6f, -2.6f}));
    scene->add(solid(BoxGeometry::create(0.9f, 0.6f, 0.9f), Color(0xe0b040), {-1.2f, 0.3f, -2.5f}));

    // Spheres resting on the floor, and one touching the stack.
    const auto sphere = SphereGeometry::create(0.5f, 64, 32);
    scene->add(solid(sphere, Color(0xf0f0f0), {0.4f, 0.5f, 0.6f}, 0.4f));
    scene->add(solid(sphere, Color(0x50a060), {1.5f, 0.5f, -0.6f}, 0.4f));
    scene->add(solid(SphereGeometry::create(0.35f, 48, 24), Color(0xa050c0), {-1.5f, 0.35f, -1.5f}, 0.4f));

    auto knot = solid(TorusKnotGeometry::create(0.38f, 0.13f, 200, 24), Color(0xd08040), {1.9f, 0.f, 1.3f}, 0.3f);
    restOn(*knot, 0.f);// standing on the floor
    scene->add(knot);

    const std::string suzannePath = std::string(DATA_FOLDER) + "/models/gltf/Suzanne/glTF/Suzanne.gltf";
    if (std::filesystem::exists(suzannePath)) {
        GLTFLoader loader;
        if (auto gltf = loader.load(suzannePath)) {
            auto model = gltf->scene;
            model->scale.setScalar(0.45f);
            model->position.set(-1.2f, 0.f, -2.4f);
            model->rotation.y = 0.5f;
            restOn(*model, 0.6f);// sitting on the yellow box
            // A plain clay material: the point here is the shape.
            auto clay = MeshStandardMaterial::create();
            clay->color = Color(0xb0a898);
            clay->roughness = 0.6f;
            model->traverseType<Mesh>([&](Mesh& m) { m.setMaterial(clay); });
            scene->add(model);
        }
    }

    EffectComposer::Options options;
    options.samples = 4;
    EffectComposer composer(renderer, options);
    composer.addPass(std::make_shared<RenderPass>(*scene, *camera));

    const auto size = canvas.size();
    GTAOParameters aoParameters;// three.js example values
    aoParameters.radius = 0.25f;
    aoParameters.distanceExponent = 1.f;
    aoParameters.thickness = 1.f;
    aoParameters.scale = 1.f;
    aoParameters.samples = 16;
    aoParameters.distanceFallOff = 1.f;
    aoParameters.screenSpaceRadius = false;
    PoissonDenoiseParameters pdParameters;
    pdParameters.lumaPhi = 10.f;
    pdParameters.depthPhi = 2.f;
    pdParameters.normalPhi = 3.f;
    pdParameters.radius = 4.f;
    pdParameters.radiusExponent = 1.f;
    pdParameters.rings = 2.f;
    pdParameters.samples = 16;

    auto gtao = std::make_shared<GTAOPass>(*scene, *camera, size.width(), size.height(), aoParameters, pdParameters);
    gtao->output = outputs[modeIndex].first;
    gtao->blendIntensity = intensity;
    gtao->enabled = !startOff;
    composer.addPass(gtao);

    canvas.onKeyPressed([&](KeyEvent evt) {
        if (evt.key == Key::G) gtao->enabled = !gtao->enabled;
        if (evt.key == Key::M) modeIndex = (modeIndex + 1) % static_cast<int>(outputs.size());
    });

    bool spin = true;

    // Every field of the two parameter structs is set above, so the sliders
    // edit them in place (&*field) and a change pushes the whole struct.
    ImguiFunctionalContext ui(canvas, [&] {
        ImGui::SetNextWindowPos({0, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({0, 0}, 0);
        ImGui::Begin("GTAO");

        ImGui::Checkbox("AO (G)", &gtao->enabled);
        if (ImGui::BeginCombo("output (M)", outputs[modeIndex].second)) {
            for (int k = 0; k < static_cast<int>(outputs.size()); k++) {
                if (ImGui::Selectable(outputs[k].second, k == modeIndex)) modeIndex = k;
            }
            ImGui::EndCombo();
        }
        ImGui::SliderFloat("blend intensity", &gtao->blendIntensity, 0.f, 1.f);
        ImGui::Checkbox("spin the knot", &spin);

        ImGui::SeparatorText("AO");
        bool ao = false;
        ao |= ImGui::SliderFloat("radius", &*aoParameters.radius, 0.01f, 1.f);
        ao |= ImGui::SliderFloat("distance exponent", &*aoParameters.distanceExponent, 1.f, 4.f);
        ao |= ImGui::SliderFloat("thickness", &*aoParameters.thickness, 0.01f, 10.f, "%.2f", ImGuiSliderFlags_Logarithmic);
        ao |= ImGui::SliderFloat("distance falloff", &*aoParameters.distanceFallOff, 0.f, 1.f);
        ao |= ImGui::SliderFloat("scale", &*aoParameters.scale, 0.01f, 2.f);
        ao |= ImGui::SliderInt("samples", &*aoParameters.samples, 2, 32);
        ao |= ImGui::Checkbox("radius in screen space", &*aoParameters.screenSpaceRadius);
        if (ao) gtao->updateGtaoMaterial(aoParameters);

        // What `samples` costs: it is rounded up to whole steps per slice
        // direction, and every step reads the depth buffer on both sides.
        const int directions = *aoParameters.samples < 30 ? 3 : 5;
        const int steps = (*aoParameters.samples + directions - 1) / directions;
        ImGui::Text("%d directions x %d steps = %d depth reads", directions, steps, 2 * directions * steps);

        ImGui::SeparatorText("Denoise");
        bool pd = false;
        pd |= ImGui::SliderFloat("luma phi", &*pdParameters.lumaPhi, 0.01f, 20.f);
        pd |= ImGui::SliderFloat("depth phi", &*pdParameters.depthPhi, 0.01f, 20.f);
        pd |= ImGui::SliderFloat("normal phi", &*pdParameters.normalPhi, 0.01f, 20.f);
        pd |= ImGui::SliderFloat("radius##pd", &*pdParameters.radius, 0.f, 32.f, "%.0f px");
        pd |= ImGui::SliderFloat("radius exponent", &*pdParameters.radiusExponent, 0.1f, 4.f);
        pd |= ImGui::SliderFloat("rings", &*pdParameters.rings, 1.f, 16.f);
        pd |= ImGui::SliderInt("samples##pd", &*pdParameters.samples, 2, 32);
        if (pd) gtao->updatePdMaterial(pdParameters);

        ImGui::End();
    });

    IOCapture ioCapture{};
    ioCapture.preventMouseEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    ioCapture.preventScrollEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    canvas.setIOCapture(&ioCapture);

    canvas.onWindowResize([&](WindowSize size) {
        if (auto* p = dynamic_cast<PerspectiveCamera*>(camera.get())) {
            p->aspect = size.aspect();
        } else if (auto* o = dynamic_cast<OrthographicCamera*>(camera.get())) {
            o->left = -o->top * size.aspect();
            o->right = o->top * size.aspect();
        }
        camera->updateProjectionMatrix();
        renderer.setSize(size);
        composer.setSize(size.width(), size.height());
    });

    int frame = 0;
    Clock clock;
    canvas.animate([&] {
        const auto dt = clock.getDelta();
        if (!capture && spin) knot->rotation.y += 0.3f * dt;

        // The AO, depth and normal outputs are data, not light, so they are
        // shown without the tone curve: an unoccluded 1 is white.
        gtao->output = outputs[modeIndex].first;
        const bool image = !gtao->enabled || gtao->output == GTAOPass::Output::Default || gtao->output == GTAOPass::Output::Diffuse;
        renderer.toneMapping = image ? ToneMapping::ACESFilmic : ToneMapping::None;

        composer.render(dt);
        if (!capture || shotWithUi) ui.render();

        // (the panel takes a few frames to settle on its size)
        if (capture && ++frame >= (shotWithUi ? 8 : 3)) {
            renderer.writeFramebuffer(shotPath);
            std::cout << "wrote " << shotPath << std::endl;
            std::exit(0);
        }
    });
}
