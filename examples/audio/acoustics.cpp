
#include "renderer_factory.hpp"

#include "threepp/threepp.hpp"

#include "threepp/audio/Acoustics.hpp"
#include "threepp/audio/Audio.hpp"
#include "threepp/extras/imgui/RendererSettings.hpp"
#include "threepp/utils/BufferGeometryUtils.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>


using namespace threepp;

namespace {

    constexpr size_t maxHitMarkers = 64;

    auto makeWallA() {

        auto mesh = Mesh::create(
                BoxGeometry::create(6, 3, 0.4f),
                MeshPhongMaterial::create(MeshPhongMaterial::Params{}.color(0x888888)));
        mesh->position.set(0, 1.5f, 3);

        return mesh;
    }

    auto makeWallB() {

        auto mesh = Mesh::create(
                BoxGeometry::create(6, 3, 0.4f),
                MeshPhongMaterial::create(MeshPhongMaterial::Params{}.color(0x2a7f62)));
        mesh->position.set(5, 1.5f, 3);
        mesh->rotation.y = math::degToRad(35);

        return mesh;
    }

    auto makeSlab(float width, float height, float depth, const Vector3& at) {

        auto mesh = Mesh::create(
                BoxGeometry::create(width, height, depth),
                MeshPhongMaterial::create(MeshPhongMaterial::Params{}.color(0x6b6f76)));
        mesh->position.copy(at);

        return mesh;
    }

    // Five slabs around an 8x4x8 volume, open towards +x. The floor is the
    // scene's ground plane.
    std::vector<std::shared_ptr<Mesh>> makeGarage(const Vector3& center) {

        return {
                makeSlab(0.3f, 4.f, 8.3f, {center.x - 4.f, 2.f, center.z}),  // back
                makeSlab(8.3f, 4.f, 0.3f, {center.x, 2.f, center.z - 4.f}),  // side
                makeSlab(8.3f, 4.f, 0.3f, {center.x, 2.f, center.z + 4.f}),  // side
                makeSlab(8.3f, 0.3f, 8.3f, {center.x, 4.f, center.z}),       // roof
                makeSlab(0.3f, 4.f, 2.6f, {center.x + 4.f, 2.f, center.z - 2.7f})};// half of the open face
    }

    // A pool of markers parked at the point where each blocked ray first met a
    // wall. This and the occlusion tint on the source meshes are the whole
    // debug visualization: the rays themselves are listener-to-source
    // sight-lines, seen exactly end-on from the listener's own camera, so any
    // attempt to draw them carries no information a color can't.
    auto makeHitMarkers() {

        auto group = Group::create();

        auto geometry = SphereGeometry::create(0.06f, 8, 6);
        auto material = MeshBasicMaterial::create(MeshBasicMaterial::Params{}.color(0xff3322));
        material->depthTest = false;

        for (size_t i = 0; i < maxHitMarkers; ++i) {
            auto marker = Mesh::create(geometry, material);
            marker->visible = false;
            marker->frustumCulled = false;
            group->add(marker);
        }

        return group;
    }

    void updateHitMarkers(Group& markers, const std::vector<AcousticDebugRay>& rays) {

        size_t marker = 0;

        for (const auto& ray : rays) {
            if (ray.firstHit && marker < markers.children.size()) {
                auto* mesh = markers.children[marker++];
                mesh->position.copy(*ray.firstHit);
                mesh->visible = true;
            }
        }

        for (; marker < markers.children.size(); ++marker) {
            markers.children[marker]->visible = false;
        }
    }

    bool check(bool ok, const std::string& what) {

        std::cout << (ok ? "  ok   " : "  FAIL ") << what << std::endl;

        return ok;
    }

    // Six slabs sealing an 8x4x8 room, for the environment probe.
    std::vector<std::shared_ptr<Mesh>> makeClosedRoom() {

        std::vector<std::shared_ptr<Mesh>> walls{
                makeSlab(8.f, 0.2f, 8.f, {0, 0, 0}),
                makeSlab(8.f, 0.2f, 8.f, {0, 4, 0}),
                makeSlab(0.2f, 4.f, 8.f, {-4, 2, 0}),
                makeSlab(0.2f, 4.f, 8.f, {4, 2, 0}),
                makeSlab(8.f, 4.f, 0.2f, {0, 2, -4}),
                makeSlab(8.f, 4.f, 0.2f, {0, 2, 4})};

        for (auto& wall : walls) wall->updateMatrixWorld();

        return walls;
    }

    int selftest() {

        bool ok = true;

        // 1) BVH ray sanity, in the geometry's own local space.
        BVH bvh;
        bvh.build(*BoxGeometry::create(1, 1, 1));

        const Ray hitRay({0, 0, -5}, {0, 0, 1});
        const auto hit = bvh.raycast(hitRay);
        ok &= check(hit.has_value(), "raycast hits a unit box head-on");
        if (hit) {
            ok &= check(std::abs(hit->distance - 4.5f) < 1e-3f,
                        "hit distance is 4.5 (got " + std::to_string(hit->distance) + ")");
        }
        ok &= check(bvh.raycastAny(hitRay, 10.f), "raycastAny agrees on the hit");

        const Ray missRay({3, 0, -5}, {0, 0, 1});
        ok &= check(!bvh.raycast(missRay).has_value(), "raycast misses when offset to the side");
        ok &= check(!bvh.raycastAny(missRay, 10.f), "raycastAny agrees on the miss");

        // 2) Occlusion through the demo's solid wall.
        auto wall = makeWallA();
        wall->updateMatrixWorld();

        AcousticScene scene;
        scene.add(*wall, AcousticSurface{0.f});

        const float blocked = scene.transmission({0, 1.5f, -2}, {0, 1.5f, 6});
        ok &= check(blocked < 0.05f, "solid wall blocks (" + std::to_string(blocked) + " < 0.05)");

        const float clear = scene.transmission({10, 1.5f, -2}, {0, 1.5f, 6});
        ok &= check(clear > 0.95f, "clear line to the side (" + std::to_string(clear) + " > 0.95)");

        // 3) A transmissive surface passes its share through.
        scene.add(*wall, AcousticSurface{0.6f});

        const float curtain = scene.transmission({0, 1.5f, -2}, {0, 1.5f, 6});
        ok &= check(curtain > 0.5f && curtain < 0.7f,
                    "curtain transmits (" + std::to_string(curtain) + " in [0.5, 0.7])");

        // 4) Hit counting: two walls attenuate twice, whether they arrive as two
        // meshes or as one mesh with two slabs in it.
        auto curtainA = makeSlab(6, 3, 0.4f, {0, 1.5f, 2});
        auto curtainB = makeSlab(6, 3, 0.4f, {0, 1.5f, 4});
        curtainA->updateMatrixWorld();
        curtainB->updateMatrixWorld();

        AcousticScene stacked;
        stacked.add(*curtainA, AcousticSurface{0.6f});
        stacked.add(*curtainB, AcousticSurface{0.6f});

        const float twoMeshes = stacked.transmission({0, 1.5f, -2}, {0, 1.5f, 6});
        ok &= check(twoMeshes > 0.3f && twoMeshes < 0.45f,
                    "two curtains as two meshes (" + std::to_string(twoMeshes) + " in [0.3, 0.45])");

        auto slabA = BoxGeometry::create(6, 3, 0.4f);
        slabA->translate(0, 0, -1);
        auto slabB = BoxGeometry::create(6, 3, 0.4f);
        slabB->translate(0, 0, 1);

        auto merged = Mesh::create(
                mergeBufferGeometries(std::vector<std::shared_ptr<BufferGeometry>>{slabA, slabB}),
                MeshBasicMaterial::create());
        merged->position.set(0, 1.5f, 3);
        merged->updateMatrixWorld();

        AcousticScene fused;
        fused.add(*merged, AcousticSurface{0.6f});

        const float oneMesh = fused.transmission({0, 1.5f, -2}, {0, 1.5f, 6});
        ok &= check(oneMesh > 0.3f && oneMesh < 0.45f,
                    "two curtains merged into one mesh (" + std::to_string(oneMesh) + " in [0.3, 0.45])");

        // 5) Environment probe.
        auto room = makeClosedRoom();

        AcousticScene closed;
        for (const auto& slab : room) closed.add(*slab, AcousticSurface{0.f, 0.05f});

        const auto live = closed.probe({0, 2, 0});
        ok &= check(live.escapeFraction < 0.05f,
                    "closed room keeps its rays (escape " + std::to_string(live.escapeFraction) + " < 0.05)");
        ok &= check(live.rt60 > 1.f,
                    "closed room rings (rt60 " + std::to_string(live.rt60) + " > 1 s)");
        ok &= check(live.meanFreePath > 1.f && live.meanFreePath < 8.f,
                    "mean free path (" + std::to_string(live.meanFreePath) + " in [1, 8] m)");

        auto ground = Mesh::create(PlaneGeometry::create(60, 60), MeshBasicMaterial::create());
        ground->rotateX(-math::PI / 2);
        ground->updateMatrixWorld();

        AcousticScene outdoors;
        outdoors.add(*ground, AcousticSurface{0.f, 0.4f});

        const auto open = outdoors.probe({0, 1.7f, 0});
        ok &= check(open.escapeFraction > 0.5f,
                    "open ground lets rays go (escape " + std::to_string(open.escapeFraction) + " > 0.5)");
        ok &= check(open.wetLevel < 0.15f,
                    "open ground stays dry (wet " + std::to_string(open.wetLevel) + " < 0.15)");

        AcousticScene damped;
        for (const auto& slab : room) damped.add(*slab, AcousticSurface{0.f, 0.6f});

        const auto soft = damped.probe({0, 2, 0});
        ok &= check(soft.rt60 < live.rt60,
                    "absorbent room decays faster (" + std::to_string(soft.rt60) +
                            " < " + std::to_string(live.rt60) + " s)");

        if (!ok) {
            std::cout << "SELFTEST FAIL" << std::endl;
            return 1;
        }

        std::cout << "SELFTEST PASS" << std::endl;
        return 0;
    }

}// namespace

int main(int argc, char** argv) {

    const std::vector<std::string> args(argv, argv + argc);
    if (std::ranges::find(args, "--selftest") != args.end()) {
        return selftest();
    }

    Canvas canvas("Acoustics demo", {{"aa", 4}});
    auto renderer = createRenderer(canvas);

    auto scene = Scene::create();
    scene->background = Color(0x14181f);

    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 200);
    camera->position.set(0, 1.7f, -5);

    OrbitControls controls{*camera, canvas};
    controls.target.set(0, 1, 6);
    controls.update();

    auto ambient = AmbientLight::create(0xffffff, 0.4f);
    scene->add(ambient);

    auto light = DirectionalLight::create(0xffffff, 0.8f);
    light->position.set(4, 10, -2);
    scene->add(light);

    auto ground = Mesh::create(
            PlaneGeometry::create(60, 60),
            MeshPhongMaterial::create(MeshPhongMaterial::Params{}.color(0x2b2f38)));
    ground->rotateX(-math::PI / 2);
    scene->add(ground);

    auto wallA = makeWallA();
    scene->add(wallA);

    auto wallB = makeWallB();
    scene->add(wallB);

    const Vector3 garageCenter(-10, 0, 4);
    auto garage = makeGarage(garageCenter);
    for (const auto& slab : garage) scene->add(slab);

    AudioListener listener;
    const std::string track = std::string(DATA_FOLDER) + "/sounds/376737_Skullbeatz___Bad_Cat_Maste.mp3";

    PositionalAudio radio(listener, track);
    radio.setDistanceModel(PositionalAudio::DistanceModel::Inverse);
    radio.setMinDistance(1);
    radio.setLooping(true);

    auto radioMaterial = MeshPhongMaterial::create(MeshPhongMaterial::Params{}.color(Color::orange));
    auto radioNode = Mesh::create(BoxGeometry::create(0.4f, 0.3f, 0.25f), radioMaterial);
    radioNode->position.set(0, 1, 6);
    radioNode->addRef(radio);
    scene->add(radioNode);

    PositionalAudio inside(listener, track);
    inside.setDistanceModel(PositionalAudio::DistanceModel::Inverse);
    inside.setMinDistance(1);
    inside.setLooping(true);
    inside.setVolume(0.7f);

    auto insideMaterial = MeshPhongMaterial::create(MeshPhongMaterial::Params{}.color(Color::orange));
    auto insideNode = Mesh::create(BoxGeometry::create(0.4f, 0.3f, 0.25f), insideMaterial);
    insideNode->position.set(garageCenter.x, 1, garageCenter.z);
    insideNode->addRef(inside);
    scene->add(insideNode);

    camera->addRef(listener);

    AcousticScene acoustics;
    acoustics.add(*wallA, AcousticSurface{0.f, 0.1f});   // solid
    acoustics.add(*wallB, AcousticSurface{0.6f, 0.6f});  // curtain
    acoustics.add(*ground, AcousticSurface{0.f, 0.4f});  // flat, so it reflects without ever occluding
    for (const auto& slab : garage) acoustics.add(*slab, AcousticSurface{0.f, 0.05f});// concrete

    AcousticsSystem system(acoustics, listener);
    system.add(radio);
    system.add(inside);


    system.occlusionOf(radio);
    system.occlusionOf(inside);

    radio.play();
    inside.play();

    auto hitMarkers = makeHitMarkers();
    scene->add(hitMarkers);

    bool acousticsEnabled = true;
    bool showMarkers = true;
    float volume = listener.getMasterVolume();
    float occlusion = 0;

    RendererSettingsUi ui(canvas, *renderer, [&] {
        const auto& env = system.environment();
        ImGui::Text("Occlusion: %.2f", occlusion);
        ImGui::Text("RT60: %.2f s", env.rt60);
        ImGui::Text("Escape: %.2f", env.escapeFraction);
        ImGui::Text("Wet: %.2f", env.wetLevel);
        ImGui::Checkbox("Acoustics", &acousticsEnabled);
        if (ImGui::IsItemEdited()) {
            system.setEnabled(acousticsEnabled);
        }
        ImGui::Checkbox("Show hit markers", &showMarkers);
        ImGui::SliderFloat("Volume", &volume, 0.f, 1.f);
        if (ImGui::IsItemEdited()) {
            listener.setMasterVolume(volume);
        }
    },
                          "Acoustics");

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer->setSize(size);
    });

    Clock clock;
    canvas.animate([&] {
        // The acoustics read matrixWorld, and the renderer only refreshes it at
        // the end of the frame — do it up front so frame 1 isn't queried against
        // identity transforms.
        scene->updateMatrixWorld();
        camera->updateMatrixWorld();

        system.update(clock.getDelta());
        occlusion = system.occlusionOf(radio);

        // At-a-glance audibility, per source.
        radioMaterial->color.lerpColors(Color(0x33dd55), Color(0xdd2222), occlusion);
        insideMaterial->color.lerpColors(Color(0x33dd55), Color(0xdd2222), system.occlusionOf(inside));

        hitMarkers->visible = showMarkers;
        if (showMarkers) {
            updateHitMarkers(*hitMarkers, system.debugRays());
        }

        renderer->render(*scene, *camera);

        ui.render();
    });
}
