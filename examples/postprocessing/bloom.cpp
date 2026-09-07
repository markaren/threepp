// UnrealBloomPass: threshold the image, blur what survives across five
// progressively halved mips, add it back weighted.
//
// The threshold is applied to the tone-mapped image, which never reaches 1.0 -
// so it sits below 1 here. The composer's targets are half float, so values
// above 1 would survive the chain; what caps them is that the RenderPass tone
// maps on the way in. A chain that wants to threshold true HDR values instead
// has to take the tone mapping off the renderer and put it at the end of the
// chain as a pass of its own.
//
//   1 / 2   strength down / up
//   3 / 4   threshold down / up
//   5 / 6   radius down / up
//   B       bypass the bloom

#include "threepp/input/KeyListener.hpp"
#include "threepp/postprocessing/postprocessing.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace threepp;

namespace {

    std::shared_ptr<Mesh> glowingSphere(const Vector3& pos, const Color& color, float intensity) {

        auto material = MeshStandardMaterial::create();
        material->color = Color(0.05f, 0.05f, 0.05f);
        material->emissive = color;
        material->emissiveIntensity = intensity;
        material->roughness = 0.4f;

        auto sphere = Mesh::create(SphereGeometry::create(0.5f, 32, 16), material);
        sphere->position.copy(pos);

        return sphere;
    }

}// namespace


int main(int argc, char** argv) {

    // Headless capture (dev): bloom --shot <name.png> [--frames N] [--strength X]
    std::string shotPath;
    int shotFrames = 30, shotFrame = 0;
    float strength = 0.9f;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shot" && i + 1 < argc) shotPath = argv[++i];
        else if (a == "--frames" && i + 1 < argc) shotFrames = std::atoi(argv[++i]);
        else if (a == "--strength" && i + 1 < argc) strength = std::stof(argv[++i]);
    }

    Canvas canvas("Bloom", {{"aa", 0}});
    GLRenderer renderer(canvas);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = 1.f;

    auto scene = Scene::create();
    scene->background = Color(0x05050a);

    auto camera = PerspectiveCamera::create(55, canvas.aspect(), 0.1f, 100.f);
    camera->position.set(0, 1.5f, 7);
    camera->lookAt(Vector3{0, 0, 0});

    OrbitControls controls{*camera, canvas};

    scene->add(AmbientLight::create(0x303042));
    auto keyLight = DirectionalLight::create(0xffffff, 1.8f);
    keyLight->position.set(3, 5, 4);
    scene->add(keyLight);

    auto emitters = Group::create();
    emitters->add(glowingSphere({-2.4f, 0, 0}, Color(0xff3311), 3.f));// luma weights differ per channel,
    emitters->add(glowingSphere({0, 0, 0}, Color(0x33ff66), 2.2f));// so equal intensities would not
    emitters->add(glowingSphere({2.4f, 0, 0}, Color(0x3366ff), 3.5f));// bloom equally - green needs least
    scene->add(emitters);

    // Something dull, so the bloom has a surface to bleed over and it is
    // obvious the glow is added to the image rather than lighting the scene.
    auto slabMaterial = MeshStandardMaterial::create();
    slabMaterial->color = Color(0x6a6a78);
    slabMaterial->roughness = 0.8f;
    auto slab = Mesh::create(BoxGeometry::create(14, 0.4f, 4), slabMaterial);
    slab->position.y = -1.2f;
    scene->add(slab);

    EffectComposer::Options options;
    options.samples = 4;

    EffectComposer composer(renderer, options);
    composer.addPass(std::make_shared<RenderPass>(*scene, *camera));

    auto bloom = std::make_shared<UnrealBloomPass>(
            Vector2(static_cast<float>(canvas.size().width()), static_cast<float>(canvas.size().height())),
            strength, 0.4f, 0.35f);
    composer.addPass(bloom);

    canvas.onKeyPressed([&](KeyEvent evt) {
        switch (evt.key) {
            case Key::NUM_1: bloom->strength = std::max(0.f, bloom->strength - 0.2f); break;
            case Key::NUM_2: bloom->strength += 0.2f; break;
            case Key::NUM_3: bloom->threshold = std::max(0.f, bloom->threshold - 0.05f); break;
            case Key::NUM_4: bloom->threshold = std::min(1.f, bloom->threshold + 0.05f); break;
            case Key::NUM_5: bloom->radius = std::max(0.f, bloom->radius - 0.1f); break;
            case Key::NUM_6: bloom->radius = std::min(1.f, bloom->radius + 0.1f); break;
            case Key::B: bloom->enabled = !bloom->enabled; break;
            default: return;
        }
        std::cout << (bloom->enabled ? "bloom" : "bloom (off)")
                  << "  strength " << bloom->strength
                  << "  threshold " << bloom->threshold
                  << "  radius " << bloom->radius << std::endl;
    });

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
        composer.setSize(size.width(), size.height());
    });

    Clock clock;
    canvas.animate([&] {
        const auto dt = clock.getDelta();

        emitters->rotation.y += 0.5f * dt;

        composer.render(dt);

        if (!shotPath.empty() && ++shotFrame >= shotFrames) {
            renderer.writeFramebuffer(shotPath);
            std::cout << "wrote " << shotPath << std::endl;
            std::exit(0);
        }
    });
}
