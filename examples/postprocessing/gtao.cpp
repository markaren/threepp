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
//   1 / 2    AO radius down / up
//   3 / 4    blend intensity down / up
//   5 / 6    samples down / up (the main cost knob)

#include "threepp/extras/environments/RoomEnvironment.hpp"
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
    //        [--off] [--ortho] [--intensity X] [--size WxH]
    std::string shotPath;
    int modeIndex = 0;
    bool startOff = false, ortho = false;
    float intensity = 1.f;
    int width = 1280, height = 720;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shot" && i + 1 < argc) shotPath = argv[++i];
        else if (a == "--off") startOff = true;
        else if (a == "--ortho") ortho = true;
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

    std::shared_ptr<Camera> camera;
    if (ortho) {
        const float h = 4.f, w = h * canvas.aspect();
        camera = OrthographicCamera::create(-w, w, h, -h, 0.1f, 50.f);
    } else {
        camera = PerspectiveCamera::create(40, canvas.aspect(), 0.1f, 50.f);
    }
    camera->position.set(3.4f, 2.7f, 5.0f);

    OrbitControls controls{*camera, canvas};
    controls.target.set(-0.8f, 0.6f, -0.8f);
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

    auto knot = solid(TorusKnotGeometry::create(0.38f, 0.13f, 200, 24), Color(0xd08040), {1.9f, 0.52f, 1.3f}, 0.3f);
    scene->add(knot);

    const std::string suzannePath = std::string(DATA_FOLDER) + "/models/gltf/Suzanne/glTF/Suzanne.gltf";
    if (std::filesystem::exists(suzannePath)) {
        GLTFLoader loader;
        if (auto gltf = loader.load(suzannePath)) {
            auto model = gltf->scene;
            model->scale.setScalar(0.45f);
            model->position.set(-1.2f, 0.95f, -2.4f);// sitting on the yellow box
            model->rotation.y = 0.5f;
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

    float radius = *aoParameters.radius;
    int samples = *aoParameters.samples;

    canvas.onKeyPressed([&](KeyEvent evt) {
        switch (evt.key) {
            case Key::G: gtao->enabled = !gtao->enabled; break;
            case Key::M:
                modeIndex = (modeIndex + 1) % static_cast<int>(outputs.size());
                gtao->output = outputs[modeIndex].first;
                break;
            case Key::NUM_1: radius = std::max(0.05f, radius - 0.05f); break;
            case Key::NUM_2: radius = std::min(2.f, radius + 0.05f); break;
            case Key::NUM_3: gtao->blendIntensity = std::max(0.f, gtao->blendIntensity - 0.1f); break;
            case Key::NUM_4: gtao->blendIntensity = std::min(1.f, gtao->blendIntensity + 0.1f); break;
            case Key::NUM_5: samples = std::max(2, samples - 2); break;
            case Key::NUM_6: samples = std::min(32, samples + 2); break;
            default: return;
        }
        GTAOParameters update;
        update.radius = radius;
        update.samples = samples;
        gtao->updateGtaoMaterial(update);

        std::cout << (gtao->enabled ? "AO on" : "AO off")
                  << "  output " << outputs[modeIndex].second
                  << "  radius " << radius
                  << "  samples " << samples
                  << "  intensity " << gtao->blendIntensity << std::endl;
    });

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
        if (!capture) knot->rotation.y += 0.3f * dt;

        composer.render(dt);

        if (capture && ++frame >= 3) {
            renderer.writeFramebuffer(shotPath);
            std::cout << "wrote " << shotPath << std::endl;
            std::exit(0);
        }
    });
}
