// BokehPass: depth of field over a receding row of objects.
//
// The pass re-renders the scene's depth into a target of its own (packed into
// RGBA, so it is point sampled), then blurs each pixel by how far its view
// depth sits from the focus plane. Focus is in world units along the view axis.
//
//   Left / Right   pull focus nearer / further
//   Up / Down      aperture up / down
//   B              bypass the depth of field

#include "threepp/input/KeyListener.hpp"
#include "threepp/postprocessing/postprocessing.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace threepp;


int main(int argc, char** argv) {

    // Headless capture (dev): dof --shot <name.png> [--frames N] [--focus X]
    std::string shotPath;
    int shotFrames = 30, shotFrame = 0;
    float focus = 7.f;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shot" && i + 1 < argc) shotPath = argv[++i];
        else if (a == "--frames" && i + 1 < argc) shotFrames = std::atoi(argv[++i]);
        else if (a == "--focus" && i + 1 < argc) focus = std::stof(argv[++i]);
    }

    Canvas canvas("Depth of field", {{"aa", 0}});
    GLRenderer renderer(canvas);

    auto scene = Scene::create();
    scene->background = Color(0x2b2b38);

    auto camera = PerspectiveCamera::create(45, canvas.aspect(), 0.5f, 60.f);
    camera->position.set(2.5f, 1.6f, 4.f);
    camera->lookAt(Vector3{0, 0.2f, -6});

    OrbitControls controls{*camera, canvas};

    scene->add(AmbientLight::create(0x555568));
    auto keyLight = DirectionalLight::create(0xffffff, 1.4f);
    keyLight->position.set(4, 8, 6);
    scene->add(keyLight);

    // A row marching away from the camera: the whole point is that only one
    // depth is sharp, so the scene has to span depth.
    auto geometry = BoxGeometry::create(0.9f, 0.9f, 0.9f);
    for (int i = 0; i < 12; ++i) {
        auto material = MeshStandardMaterial::create();
        material->color = Color().setHSL(static_cast<float>(i) / 12.f, 0.6f, 0.55f);
        material->roughness = 0.55f;

        auto box = Mesh::create(geometry, material);
        box->position.set((i % 2 == 0) ? -0.9f : 0.9f, 0, -static_cast<float>(i) * 1.9f);
        box->rotation.y = static_cast<float>(i) * 0.3f;
        scene->add(box);
    }

    auto floorMaterial = MeshStandardMaterial::create();
    floorMaterial->color = Color(0x40404e);
    floorMaterial->roughness = 0.9f;
    auto floor = Mesh::create(PlaneGeometry::create(40, 60), floorMaterial);
    floor->rotateX(-math::PI / 2);
    floor->position.set(0, -0.46f, -10);
    scene->add(floor);

    EffectComposer::Options options;
    options.samples = 4;

    EffectComposer composer(renderer, options);
    composer.addPass(std::make_shared<RenderPass>(*scene, *camera));

    auto bokeh = std::make_shared<BokehPass>(*scene, *camera, focus, 0.014f, 0.009f);
    composer.addPass(bokeh);

    canvas.onKeyPressed([&](KeyEvent evt) {
        switch (evt.key) {
            case Key::LEFT: bokeh->focus = std::max(0.5f, bokeh->focus - 0.8f); break;
            case Key::RIGHT: bokeh->focus += 0.8f; break;
            case Key::UP: bokeh->aperture += 0.005f; break;
            case Key::DOWN: bokeh->aperture = std::max(0.f, bokeh->aperture - 0.005f); break;
            case Key::B: bokeh->enabled = !bokeh->enabled; break;
            default: return;
        }
        std::cout << (bokeh->enabled ? "dof" : "dof (off)")
                  << "  focus " << bokeh->focus
                  << "  aperture " << bokeh->aperture << std::endl;
    });

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
        composer.setSize(size.width(), size.height());
    });

    Clock clock;
    canvas.animate([&] {
        composer.render(clock.getDelta());

        if (!shotPath.empty() && ++shotFrame >= shotFrames) {
            renderer.writeFramebuffer(shotPath);
            std::cout << "wrote " << shotPath << std::endl;
            std::exit(0);
        }
    });
}
