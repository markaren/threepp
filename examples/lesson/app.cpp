// app.cpp: the program built in "A threepp lesson, Part 0b: your first real app".
//
// It starts as hello.cpp from Part 0 and grows one step at a time. Each step is
// marked with `#if STEP >= n`, so building with STEP=n (in CMake: -DLESSON_APP_STEP=n)
// gives the program as it stands after step n (0 is hello.cpp). The default is the
// whole app:
//
//   1  a camera you can move          OrbitControls, and a window that resizes
//   2  a real model                   GLTFLoader, in place of the box
//   3  light from a photo             an HDR sky as background and environment
//   4  motion                         AnimationMixer, and the clock's delta time
//   5  shadows                        which objects cast them, which receive them
//   6  many things, one draw call     InstancedMesh
//   7  point at things                Raycaster, from the mouse into the scene
//   8  knobs                          a Dear ImGui panel
//
// LESSON_CAPTURE is only for the film: capture.hpp runs this same source headless,
// on a fixed time step, with a scripted mouse, and writes the frames.

#include "threepp/threepp.hpp"

#include "threepp/animation/AnimationMixer.hpp"
#include "threepp/core/Raycaster.hpp"
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/loaders/RGBELoader.hpp"

#include <cmath>
#include <filesystem>
#include <random>

#ifndef STEP
#define STEP 99
#endif
#ifdef LESSON_CAPTURE
#include "capture.hpp"
#endif

using namespace threepp;

int main() {
    const std::filesystem::path data = DATA_FOLDER;

    Canvas canvas("app", {{"size", WindowSize{1280, 720}}});
    GLRenderer renderer(canvas);

    Scene scene;

    PerspectiveCamera camera(60, canvas.aspect(), 0.1f, 100);
    camera.position.set(0, 1.5f, 4);
    camera.lookAt(0, 0, 0);

#if STEP >= 1
    OrbitControls controls(camera, canvas);// drag to turn, scroll to zoom
    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(size);
    });
#endif

#if STEP < 2
    auto material = MeshStandardMaterial::create();
    material->color = 0x4cc9f0;
    auto box = Mesh::create(BoxGeometry::create(), material);
    scene.add(box);
#else
    GLTFLoader loader;
    auto gltf = loader.load(data / "models/gltf/Soldier.glb");
    auto soldier = gltf->scene;
    soldier->rotation.y = math::PI;// turn him to face the camera
    scene.add(soldier);

    auto groundMaterial = MeshStandardMaterial::create();
    groundMaterial->color = 0x4f5443;
    auto ground = Mesh::create(CircleGeometry::create(8, 96), groundMaterial);
    ground->rotation.x = -math::PI / 2;
    scene.add(ground);

    camera.position.set(0, 1.6f, 3.6f);
    controls.target.set(0, 0.9f, 0);
    controls.update();
#endif

    auto light = DirectionalLight::create(0xffffff, 3.f);
    light->position.set(3, 6, 4);
    scene.add(light);

#if STEP >= 3
    RGBELoader hdrLoader;
    auto sky = hdrLoader.load(data / "textures/env/noon_grass_2k.hdr");
    scene.background = sky; // what you see behind everything
    scene.environment = sky;// what every material reflects
    renderer.toneMapping = ToneMapping::ACESFilmic;
#endif

#if STEP >= 4
    AnimationMixer mixer(*soldier);
    auto clip = AnimationClip::findByName(gltf->animations, "Walk");
    mixer.clipAction(clip)->play();
#endif

#if STEP >= 5
    renderer.shadowMap().enabled = true;
    light->castShadow = true;
    soldier->traverseType<Mesh>([](Mesh& m) { m.castShadow = true; });
    ground->receiveShadow = true;

    // the light's own camera: only what is inside it gets a shadow
    auto* shadowCamera = light->shadow->camera->as<OrthographicCamera>();
    shadowCamera->left = shadowCamera->bottom = -7;
    shadowCamera->right = shadowCamera->top = 7;
    shadowCamera->updateProjectionMatrix();
#endif

#if STEP >= 6
    const int count = 2000;
    auto stoneGeometry = IcosahedronGeometry::create(0.1f);
    auto stoneMaterial = MeshStandardMaterial::create();
    auto stones = InstancedMesh::create(stoneGeometry, stoneMaterial, count);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(0, 1);
    Matrix4 matrix;
    Color color;
    for (int i = 0; i < count; i++) {
        const float r = 1.2f + 5.5f * std::sqrt(u(rng));// around him
        const float a = 2 * math::PI * u(rng);
        const float s = 0.5f + u(rng);
        const float tiltX = 3 * u(rng), tiltY = 3 * u(rng);
        Quaternion q;
        q.setFromEuler(Euler(tiltX, tiltY, 0));
        matrix.compose({r * std::cos(a), 0.05f * s, r * std::sin(a)}, q, {s, s, s});
        stones->setMatrixAt(i, matrix);
        stones->setColorAt(i, color.setHSL(0.08f, 0.15f, 0.12f + 0.2f * u(rng)));
#ifdef SEPARATE_MESHES
        auto material = MeshStandardMaterial::create();// one mesh per stone
        material->color = color;
        auto stone = Mesh::create(stoneGeometry, material);
        matrix.decompose(stone->position, stone->quaternion, stone->scale);
        stone->castShadow = stone->receiveShadow = true;
        scene.add(stone);
#endif
    }
    stones->castShadow = true;
    stones->receiveShadow = true;
#ifndef SEPARATE_MESHES
    scene.add(stones);
#endif
#endif

#if STEP >= 7
    Raycaster raycaster;
    Vector2 mouse{-10, -10};// off screen until the mouse moves
    MouseMoveListener onMove([&](const Vector2& pos) {
        auto size = canvas.size();
        mouse.x = pos.x / size.width() * 2 - 1;// pixels to -1..1
        mouse.y = -pos.y / size.height() * 2 + 1;
    });
    canvas.addMouseListener(onMove);
    int hovered = -1;
    Color hoveredColor;
#endif

#if STEP >= 8
    float sunAngle = 53, speed = 1;
    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({20, 20}, ImGuiCond_Once);
        ImGui::Begin("app", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::SliderFloat("sun", &sunAngle, 0, 360);
        ImGui::SliderFloat("speed", &speed, 0, 2);
        ImGui::End();
    });
    IOCapture capture;// the camera ignores the mouse while it is on the panel
    capture.preventMouseEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    canvas.setIOCapture(&capture);
#endif

    Clock clock;
    canvas.animate([&] {
        const float dt = clock.getDelta();

#if STEP < 2
        box->rotation.y += dt;
#endif
#if STEP >= 8
        const float a = math::degToRad(sunAngle);
        light->position.set(5 * std::cos(a), 6, 5 * std::sin(a));
        mixer.update(dt * speed);
#elif STEP >= 4
        mixer.update(dt);
#endif
#if STEP >= 7
        raycaster.setFromCamera(mouse, camera);
        auto hits = raycaster.intersectObject(*stones);
        int hit = hits.empty() ? -1 : *hits.front().instanceId;
        if (hit != hovered) {
            if (hovered >= 0) stones->setColorAt(hovered, hoveredColor);
            if (hit >= 0) {
                stones->getColorAt(hit, hoveredColor);// remember it
                stones->setColorAt(hit, Color(0xffb347));
            }
            stones->instanceColor()->needsUpdate();
            hovered = hit;
        }
#endif
        renderer.render(scene, camera);
#if STEP >= 8
        ui.render();
#endif
    });
}
