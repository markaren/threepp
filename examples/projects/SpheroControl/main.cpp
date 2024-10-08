
#include "threepp/helpers/CameraHelper.hpp"

#include "threepp/threepp.hpp"

#include "Sphero.hpp"
#include "KeyController.hpp"

#include "threepp/renderers/GLRenderTarget.hpp"
#include "threepp/utils/ImageUtils.hpp"

#include <iostream>

using namespace threepp;

namespace {

    auto createSprite(const std::shared_ptr<Texture> &texture) {
        auto spriteMaterial = MeshBasicMaterial::create({{"map", texture}});
        auto sprite = Mesh::create(PlaneGeometry::create(), spriteMaterial);
        sprite->scale.set(2, 2, 1);

        return sprite;
    }

    void addWalls(Scene &scene) {
        auto wallMaterial = MeshStandardMaterial::create({{"color", Color::black}});
        auto wall1 = Mesh::create(BoxGeometry::create(0.1, 1, 10), wallMaterial);
        wall1->position.set(-5, 0.5, 0);
        scene.add(wall1);
        auto wall2 = wall1->clone();
        wall2->translateX(10);
        scene.add(wall2);
    }

    void setupScene(Scene &scene) {

        auto light1 = HemisphereLight::create();
        light1->intensity = 0.8f;
        scene.add(light1);

        auto grid = GridHelper::create(100, 100);
        scene.add(grid);

        addWalls(scene);

        auto sphereMaterial = MeshBasicMaterial::create({{"color", Color::blue}});
        auto sphere1 = Mesh::create(SphereGeometry::create(2.f), sphereMaterial);
        sphere1->name = "target";
        sphere1->position.set(0, 2, 50);
        scene.add(sphere1);
    }

}// namespace

int main() {

    constexpr unsigned int textureSize = 256;

    Canvas canvas("Sphero simulator", {{"aa", 8}});
    auto size = canvas.size();
    GLRenderer renderer(size);
    renderer.autoClear = false;

    Scene scene;
    scene.background = Color::aliceblue;

    PerspectiveCamera camera(60, canvas.aspect(), 0.1f, 1000);
    camera.position.set(0, 5, -10);
    camera.layers.enableAll();

    setupScene(scene);

    Sphero sphero;
    sphero.position.y = 0.25;
    scene.add(sphero);

    camera.lookAt(sphero.position);
    sphero.add(camera);

    auto &spheroCamera = sphero.camera();
    auto cameraHelper = CameraHelper::create(spheroCamera);
    scene.add(cameraHelper);

    KeyController keyController(sphero);
    canvas.addKeyListener(keyController);
    std::cout << "Press 'r' to change driving mode. " << std::endl;

    OrbitControls controls(camera, canvas);
    controls.enableKeys = false;
    controls.enableZoom = false;
    controls.enablePan = false;

    Scene orthoScene;
    OrthographicCamera orthoCamera(-1, 1, 1, -1, 1, 10);
    orthoCamera.position.z = 1;

    GLRenderTarget::Options opts;
    opts.format = Format::RGB;
    opts.anisotropy = 16;
    GLRenderTarget renderTarget(textureSize, textureSize, opts);
    orthoScene.add(createSprite(renderTarget.texture));

    canvas.onWindowResize([&](WindowSize newSize) {
        camera.aspect = newSize.aspect();
        camera.updateProjectionMatrix();

        renderer.setSize(newSize);

        size = newSize;
    });

    Clock clock1, clock2;
    float imageRefreshInterval = 0.1;
    auto target = scene.getObjectByName("target");
    canvas.animate([&] {
        const auto dt = clock1.getDelta();

        keyController.update();
        sphero.update(dt);

        controls.target = sphero.position;
        controls.update();

        target->position.x = 25 * std::sin(math::TWO_PI * 0.05f * clock1.elapsedTime);

        renderer.clear();
        cameraHelper->visible = false;
        renderer.setRenderTarget(&renderTarget);
        renderer.render(scene, spheroCamera);
        renderer.setRenderTarget(nullptr);

        renderer.clear();
        cameraHelper->visible = true;
        renderer.render(scene, camera);

        renderer.clearDepth();
        renderer.setViewport({0, 0}, {textureSize, textureSize});
        renderer.render(orthoScene, orthoCamera);
        renderer.setViewport({0, 0}, size);

        if (clock2.getElapsedTime() > imageRefreshInterval) {

            renderer.copyTextureToImage(*renderTarget.texture);
            const auto pixels = renderTarget.texture->image().data();
            // Pixels can be used for stuff like OpenCV

            clock2.start();
        }
    });

}
