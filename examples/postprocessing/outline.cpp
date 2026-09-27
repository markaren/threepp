// OutlinePass: click an object to outline it, the way an editor shows what is
// selected. The outline stays visible through whatever is in front of the
// selection, in a second colour - move the camera so the ball goes behind the
// pillar to see it.
//
//   left click   select what is under the cursor (empty space clears)
//   shift+click  add to / remove from the selection
//   1 / 2        edge strength down / up
//   3 / 4        edge thickness down / up
//   5 / 6        edge glow down / up
//   P            toggle a pulse
//   T            toggle the pattern fill
//   O            bypass the outline

#include "threepp/animation/AnimationMixer.hpp"
#include "threepp/geometries/OctahedronGeometry.hpp"
#include "threepp/geometries/TorusKnotGeometry.hpp"
#include "threepp/input/KeyListener.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/postprocessing/postprocessing.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>

using namespace threepp;

namespace {

    std::shared_ptr<Mesh> shape(const std::string& name,
                                const std::shared_ptr<BufferGeometry>& geometry,
                                const Color& color,
                                const Vector3& position) {

        auto material = MeshStandardMaterial::create();
        material->color = color;
        material->roughness = 0.55f;

        auto mesh = Mesh::create(geometry, material);
        mesh->name = name;
        mesh->position.copy(position);
        mesh->castShadow = true;
        mesh->receiveShadow = true;

        return mesh;
    }

    // A click selects the whole top-level object the hit mesh belongs to, so
    // clicking the soldier's arm outlines the soldier.
    Object3D* selectable(Object3D* hit, Scene& scene) {

        while (hit && hit->parent && hit->parent != &scene) hit = hit->parent;
        return hit;
    }

    class ClickListener: public MouseListener {

    public:
        std::optional<Vector2> clicked;
        Vector2 downPos;

        void onMouseDown(int button, const Vector2& pos) override {
            if (button == 0) downPos = pos;
        }

        void onMouseUp(int button, const Vector2& pos) override {
            // A drag is the orbit controls' business, not a selection.
            if (button == 0 && pos.distanceTo(downPos) < 4.f) clicked = pos;
        }
    };

}// namespace


int main(int argc, char** argv) {

    // Headless capture (dev):
    //   outline --shot <name.png> [--frames N] [--select ball,cubes,...] [--size WxH]
    //           [--pattern] [--glow X]
    std::string shotPath;
    std::string selectNames = "ball";
    int shotFrames = 10, shotFrame = 0;
    bool startPattern = false;
    float startGlow = 0.f;
    int width = 1280, height = 720;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shot" && i + 1 < argc) shotPath = argv[++i];
        else if (a == "--frames" && i + 1 < argc) shotFrames = std::atoi(argv[++i]);
        else if (a == "--select" && i + 1 < argc) selectNames = argv[++i];
        else if (a == "--pattern") startPattern = true;
        else if (a == "--glow" && i + 1 < argc) startGlow = std::stof(argv[++i]);
        else if (a == "--size" && i + 1 < argc) std::sscanf(argv[++i], "%dx%d", &width, &height);
    }
    const bool capture = !shotPath.empty();

    Canvas canvas(Canvas::Parameters()
                          .title("Outline")
                          .size(width, height)
                          .antialiasing(0)
                          .headless(capture));
    GLRenderer renderer(canvas);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.shadowMap().enabled = true;
    renderer.shadowMap().type = ShadowMap::PFC;

    auto scene = Scene::create();
    scene->background = Color(0x9fb4c8);

    auto camera = PerspectiveCamera::create(45, canvas.aspect(), 0.1f, 100.f);
    camera->position.set(0, 3.2f, 8.5f);

    OrbitControls controls{*camera, canvas};
    controls.target.set(0, 0.8f, 0);
    controls.update();

    scene->add(HemisphereLight::create(0xddeeff, 0x554433, 0.8f));
    auto sun = DirectionalLight::create(0xffffff, 2.2f);
    sun->position.set(4, 8, 5);
    sun->castShadow = true;
    sun->shadow->mapSize.set(2048, 2048);
    auto& shadowCamera = dynamic_cast<OrthographicCamera&>(*sun->shadow->camera);
    shadowCamera.left = shadowCamera.bottom = -7;
    shadowCamera.right = shadowCamera.top = 7;
    shadowCamera.updateProjectionMatrix();
    scene->add(sun);

    auto floorMaterial = MeshStandardMaterial::create();
    floorMaterial->color = Color(0x8a8f96);
    auto floor = Mesh::create(PlaneGeometry::create(20, 20), floorMaterial);
    floor->name = "floor";
    floor->rotation.x = -math::PI / 2;
    floor->receiveShadow = true;
    scene->add(floor);

    // A ring of shapes, and a pillar standing in front of the ball so part of
    // the ball's outline is hidden from the default view.
    scene->add(shape("ball", SphereGeometry::create(0.8f, 48, 24), Color(0xd04030), {-1.1f, 0.8f, -0.6f}));
    scene->add(shape("pillar", BoxGeometry::create(0.45f, 2.4f, 0.45f), Color(0xe0d8c8), {-0.8f, 1.2f, 1.6f}));
    scene->add(shape("box", BoxGeometry::create(1, 1, 1), Color(0x3070d0), {1.3f, 0.5f, 0.4f}));
    scene->add(shape("knot", TorusKnotGeometry::create(0.5f, 0.17f, 128, 16), Color(0xe0a020), {2.9f, 1.0f, -1.6f}));
    scene->add(shape("cone", ConeGeometry::create(0.6f, 1.4f, 32), Color(0x40a060), {-3.0f, 0.7f, 0.5f}));
    scene->add(shape("cylinder", CylinderGeometry::create(0.45f, 0.45f, 1.2f, 32), Color(0x9050c0), {0.3f, 0.6f, -2.6f}));
    scene->add(shape("ico", IcosahedronGeometry::create(0.55f, 0), Color(0x20b0b0), {3.2f, 0.55f, 1.4f}));
    scene->add(shape("torus", TorusGeometry::create(0.55f, 0.2f, 24, 64), Color(0xd060a0), {-2.6f, 0.75f, -2.4f}));
    scene->add(shape("octa", OctahedronGeometry::create(0.6f), Color(0xc0c040), {-3.4f, 0.5f, 2.6f}));

    // Instanced: one draw call, one selectable object, five cubes.
    {
        auto material = MeshStandardMaterial::create();
        material->color = Color(0x607080);
        auto cubes = InstancedMesh::create(BoxGeometry::create(0.4f, 0.4f, 0.4f), material, 5);
        cubes->name = "cubes";
        Matrix4 m;
        for (int i = 0; i < 5; i++) {
            m.makeTranslation(-1.6f + 0.8f * static_cast<float>(i), 0.2f, 3.0f);
            cubes->setMatrixAt(i, m);
        }
        cubes->castShadow = true;
        cubes->receiveShadow = true;
        scene->add(cubes);
    }

    // Skinned: the soldier walks in place, and his outline follows the pose.
    std::unique_ptr<AnimationMixer> mixer;
    std::shared_ptr<Group> soldier;
    const std::string soldierPath = std::string(DATA_FOLDER) + "/models/gltf/Soldier.glb";
    if (std::filesystem::exists(soldierPath)) {
        GLTFLoader loader;
        if (auto gltf = loader.load(soldierPath)) {
            soldier = gltf->scene;
            soldier->name = "soldier";
            soldier->position.set(1.6f, 0, -3.4f);
            soldier->rotation.y = math::PI;
            soldier->scale.setScalar(1.3f);
            soldier->traverseType<Mesh>([](Mesh& m) { m.castShadow = true; });
            scene->add(soldier);
            if (!gltf->animations.empty()) {
                mixer = std::make_unique<AnimationMixer>(*soldier);
                for (auto& clip : gltf->animations) {
                    if (clip->name() == "Walk") mixer->clipAction(clip)->play();
                }
            }
        }
    }

    EffectComposer::Options options;
    options.samples = 4;
    EffectComposer composer(renderer, options);
    composer.addPass(std::make_shared<RenderPass>(*scene, *camera));

    const auto size = canvas.size();
    auto outline = std::make_shared<OutlinePass>(
            Vector2(static_cast<float>(size.width()), static_cast<float>(size.height())), *scene, *camera);
    outline->hiddenEdgeColor = Color(0.05f, 0.15f, 0.6f);// blue behind things, white in the open
    outline->usePatternTexture = startPattern;
    outline->edgeGlow = startGlow;
    composer.addPass(outline);

    TextureLoader textureLoader;
    if (auto pattern = textureLoader.load(std::string(DATA_FOLDER) + "/textures/checker.png")) {
        pattern->wrapS = TextureWrapping::Repeat;
        pattern->wrapT = TextureWrapping::Repeat;
        outline->patternTexture = pattern;
    }

    // Initial selection by name (capture uses this; interactively it is the ball).
    {
        std::stringstream ss(selectNames);
        std::string name;
        while (std::getline(ss, name, ',')) {
            if (auto* o = scene->getObjectByName(name)) outline->selectedObjects.push_back(o);
        }
    }

    ClickListener clicks;
    canvas.addMouseListener(clicks);
    Raycaster raycaster;

    canvas.onKeyPressed([&](KeyEvent evt) {
        switch (evt.key) {
            case Key::NUM_1: outline->edgeStrength = std::max(0.f, outline->edgeStrength - 0.5f); break;
            case Key::NUM_2: outline->edgeStrength += 0.5f; break;
            case Key::NUM_3: outline->edgeThickness = std::max(1.f, outline->edgeThickness - 0.5f); break;
            case Key::NUM_4: outline->edgeThickness = std::min(4.f, outline->edgeThickness + 0.5f); break;
            case Key::NUM_5: outline->edgeGlow = std::max(0.f, outline->edgeGlow - 0.25f); break;
            case Key::NUM_6: outline->edgeGlow = std::min(1.f, outline->edgeGlow + 0.25f); break;
            case Key::P: outline->pulsePeriod = outline->pulsePeriod > 0 ? 0.f : 2.f; break;
            case Key::T: outline->usePatternTexture = !outline->usePatternTexture; break;
            case Key::O: outline->enabled = !outline->enabled; break;
            default: return;
        }
        std::cout << (outline->enabled ? "outline" : "outline (off)")
                  << "  strength " << outline->edgeStrength
                  << "  thickness " << outline->edgeThickness
                  << "  glow " << outline->edgeGlow
                  << "  pulse " << outline->pulsePeriod
                  << "  pattern " << outline->usePatternTexture << std::endl;
    });

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
        composer.setSize(size.width(), size.height());
    });

    Clock clock;
    canvas.animate([&] {
        const auto dt = capture ? 1.f / 60.f : clock.getDelta();
        if (mixer) mixer->update(dt);

        if (clicks.clicked) {
            const auto s = canvas.size();
            Vector2 ndc{clicks.clicked->x / static_cast<float>(s.width()) * 2 - 1,
                        -clicks.clicked->y / static_cast<float>(s.height()) * 2 + 1};
            clicks.clicked.reset();

            raycaster.setFromCamera(ndc, *camera);
            auto hits = raycaster.intersectObjects(scene->children, true);
            Object3D* picked = nullptr;
            for (auto& hit : hits) {
                if (hit.object->name == "floor") continue;
                picked = selectable(hit.object, *scene);
                break;
            }

            auto& selected = outline->selectedObjects;
            const bool shiftHeld = canvas.isKeyDown(Key::LEFT_SHIFT);
            if (!shiftHeld) selected.clear();
            if (picked) {
                auto it = std::find(selected.begin(), selected.end(), picked);
                if (it != selected.end()) selected.erase(it);
                else selected.push_back(picked);
                std::cout << "selected " << picked->name << " (" << selected.size() << " total)" << std::endl;
            }
        }

        composer.render(dt);

        if (capture && ++shotFrame >= shotFrames) {
            renderer.writeFramebuffer(shotPath);
            std::cout << "wrote " << shotPath << std::endl;
            std::exit(0);
        }
    });
}
