// A simulated lidar on a tripod scans a small yard: a parked car, a man walking
// past, a stone wall and some rocks under an open sky. Its returns are drawn as a
// point cloud, coloured by height, by range or by the strength of each return.
//
// The parts to read:
//   - LidarSensor takes a real sensor's beam pattern (LidarModel::OS0_128() and the
//     others in the panel) and is placed like any other Object3D.
//   - scan() renders the scene from the sensor and gives one LidarReturn per hit
//     (position, normal, range and intensity), stamped with the sim time.
//   - The cloud is drawn on a layer of its own. The sensor's cameras see layer 0
//     only, so it never scans its own points.
//
// The sensor also runs under VulkanRenderer, where it traces its beams: swap the
// renderer in main() to try it.
//
// A still without a window: lidar --out shot.png [--returns] [--frames N]
//                                  [--cam x,y,z] [--look x,y,z]

#include "capture_util.hpp"

#include "threepp/animation/AnimationMixer.hpp"
#include "threepp/extras/imgui/RendererSettings.hpp"
#include "threepp/helpers/LidarSensor.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <optional>

using namespace threepp;

namespace {

    const std::string data = DATA_FOLDER;

    // The sensors in the panel. Each sees 360 degrees round, from 0.3 m out to 40 m.
    struct SensorChoice {
        const char* name;
        std::function<std::unique_ptr<LidarSensor>()> make;
    };
    const std::array<SensorChoice, 5> sensors{{
            {"Ouster OS0-128", [] { return std::make_unique<LidarSensor>(LidarModel::OS0_128(), 512, 0.3f, 40.f); }},
            {"Ouster OS1-64", [] { return std::make_unique<LidarSensor>(LidarModel::OS1_64(), 512, 0.3f, 40.f); }},
            {"Velodyne HDL-32E", [] { return std::make_unique<LidarSensor>(LidarModel::HDL32E(), 512, 0.3f, 40.f); }},
            {"Velodyne VLP-16", [] { return std::make_unique<LidarSensor>(LidarModel::VLP16(), 512, 0.3f, 40.f); }},
            {"Dense cube grid", [] { return std::make_unique<LidarSensor>(64u, 0.3f, 40.f); }},
    }};

    // A Poly Haven material from threepp_data/textures/pbr: colour, normal (OpenGL
    // convention) and ARM, whose red and green channels hold ambient occlusion and
    // roughness. The textures repeat (u, v) times over the surface.
    std::shared_ptr<MeshStandardMaterial> polyHaven(const std::string& set, float u, float v) {
        TextureLoader loader;
        const auto path = data + "/textures/pbr/" + set + "/" + set;
        auto load = [&](const std::string& suffix) {
            auto texture = loader.load(path + suffix);
            texture->wrapS = texture->wrapT = TextureWrapping::Repeat;
            texture->repeat.set(u, v);
            return texture;
        };
        auto material = MeshStandardMaterial::create();
        material->map = load("_diff_1k.jpg");
        material->map->colorSpace = ColorSpace::sRGB;
        material->normalMap = load("_nor_gl_1k.jpg");
        material->aoMap = material->roughnessMap = load("_arm_512.jpg");
        return material;
    }

    // The yard: forest floor, a dry-stone wall and Poly Haven's set of seven mossy rocks.
    void buildYard(Scene& scene) {
        auto ground = Mesh::create(PlaneGeometry::create(400, 400), polyHaven("forrest_ground_01", 130, 130));
        ground->rotation.x = -math::PI / 2;
        ground->receiveShadow = true;
        scene.add(ground);

        auto wall = Mesh::create(BoxGeometry::create(9, 1.1f, 0.6f), polyHaven("rock_wall_10", 3, 0.4f));
        wall->position.set(-2.f, 0.55f, -6.f);
        wall->rotation.y = 0.15f;
        wall->castShadow = wall->receiveShadow = true;
        scene.add(wall);

        GLTFLoader loader;
        if (auto rocks = loader.load(data + "/models/gltf/rock_moss_set_02/rock_moss_set_02_1k.gltf")) {
            const std::array<Vector3, 7> spots{{{-6.5f, 0, 1.5f}, {-4.f, 0, 5.5f}, {5.5f, 0, 5.f}, {7.5f, 0, -1.f},
                                                {-7.5f, 0, -3.5f}, {1.f, 0, 7.5f}, {3.f, 0, -6.5f}}};
            auto& set = *rocks->scene;
            for (size_t i = 0; i < set.children.size() && i < spots.size(); ++i) {
                auto* rock = set.children[i];
                rock->position.copy(spots[i]);
                rock->rotation.y = 2.1f * static_cast<float>(i);
                rock->scale.setScalar(1.2f + 0.15f * static_cast<float>(i % 3));
            }
            set.traverseType<Mesh>([](Mesh& m) { m.castShadow = m.receiveShadow = true; });
            scene.add(rocks->scene);
        }
    }

    // A 1967 Shelby GT500. The model comes about a hundred times too small, so scale
    // it to the real car's 4.6 m and stand it on the ground.
    void addCar(Scene& scene) {
        GLTFLoader loader;
        auto car = loader.load(data + "/models/gltf/ford_mustang_1967/1967_ford_mustang_shelby_cobra_gt500.glb");
        if (!car) return;
        auto& model = *car->scene;
        Box3 box;
        Vector3 size;
        model.updateMatrixWorld(true);
        box.setFromObject(model);
        box.getSize(size);
        model.scale.multiplyScalar(4.6f / std::max(size.x, size.z));
        model.updateMatrixWorld(true);
        box.setFromObject(model);
        model.position.set(4.6f, -box.min().y, -1.2f);
        model.rotation.y = -0.4f;
        model.traverseType<Mesh>([](Mesh& m) { m.castShadow = m.receiveShadow = true; });
        scene.add(car->scene);
    }

    // A man walking a loop round the sensor, between it and the car: three.js's
    // Soldier model and its Walk clip.
    struct Walker {
        std::shared_ptr<Group> model;
        std::unique_ptr<AnimationMixer> mixer;

        void update(float time, float dt) const {
            constexpr float radius = 3.f, speed = 1.2f;// m, m/s
            const float a = time * speed / radius;
            model->position.set(radius * std::cos(a), 0, radius * std::sin(a));
            model->rotation.y = math::PI - a;// facing the way he walks
            mixer->update(dt);
        }
    };

    std::optional<Walker> addWalker(Scene& scene) {
        GLTFLoader loader;
        auto soldier = loader.load(data + "/models/gltf/Soldier.glb");
        if (!soldier) return std::nullopt;
        Walker walker{soldier->scene, std::make_unique<AnimationMixer>(*soldier->scene)};
        for (const auto& clip : soldier->animations) {
            if (clip->name() == "Walk") walker.mixer->clipAction(clip)->play();
        }
        soldier->scene->traverseType<Mesh>([](Mesh& m) { m.castShadow = true; });
        scene.add(soldier->scene);
        return walker;
    }

    // A survey tripod with the sensor head on top at `height`. The real sensors look
    // no more than 45 degrees down, which stops short of the legs, and the head sits
    // inside the 0.3 m near limit, so neither shows in their scans.
    std::shared_ptr<Group> makeTripod(float height) {
        auto tripod = Group::create();

        auto leg = MeshStandardMaterial::create();
        leg->color = Color(0x2b2d30);
        leg->metalness = 0.6f;
        leg->roughness = 0.4f;
        const Vector3 top(0, height - 0.1f, 0);
        for (int i = 0; i < 3; ++i) {
            const float a = 2.f * math::PI * static_cast<float>(i) / 3.f;
            const Vector3 foot(0.5f * std::cos(a), 0, 0.5f * std::sin(a));
            Vector3 along;
            along.subVectors(top, foot);
            auto mesh = Mesh::create(CylinderGeometry::create(0.012f, 0.018f, along.length(), 8), leg);
            mesh->position.addVectors(top, foot).multiplyScalar(0.5f);
            mesh->quaternion.setFromUnitVectors(Vector3(0, 1, 0), along.normalize());
            mesh->castShadow = true;
            tripod->add(mesh);
        }

        // The head: an anodised can with a black window band round its middle.
        auto can = MeshStandardMaterial::create();
        can->color = Color(0xc9ccd1);
        can->metalness = 0.8f;
        can->roughness = 0.3f;
        auto window = MeshStandardMaterial::create();
        window->color = Color(0x0b0c0e);
        window->roughness = 0.15f;
        auto body = Mesh::create(CylinderGeometry::create(0.043f, 0.043f, 0.075f, 40), can);
        auto band = Mesh::create(CylinderGeometry::create(0.0436f, 0.0436f, 0.03f, 40), window);
        body->position.y = band->position.y = height;
        body->castShadow = true;
        tripod->add(body);
        tripod->add(band);
        return tripod;
    }

    // Room for the densest sensor in the panel (OS0-128: 128 beams x 1024 columns).
    constexpr size_t kMaxReturns = 1 << 18;

    std::shared_ptr<Points> makeCloud() {
        auto geometry = BufferGeometry::create();
        for (const auto* name : {"position", "color"}) {
            geometry->setAttribute(name, FloatBufferAttribute::create(std::vector<float>(kMaxReturns * 3), 3));
            geometry->getAttribute<float>(name)->setUsage(DrawUsage::Dynamic);
        }
        auto cloud = Points::create(geometry, PointsMaterial::create(PointsMaterial::Params{}.size(0.04f).vertexColors(true)));
        cloud->frustumCulled = false;// its extent changes with every scan
        cloud->layers.set(1);
        return cloud;
    }

    enum class ColourBy { Height, Range, Intensity };

    // Copy a scan into the cloud, each return coloured on a blue-to-red ramp.
    void showReturns(Points& cloud, const std::vector<LidarReturn>& returns, ColourBy by, float range) {
        auto& geometry = *cloud.geometry();
        auto& position = *geometry.getAttribute<float>("position");
        auto& colour = *geometry.getAttribute<float>("color");
        const int n = static_cast<int>(std::min(returns.size(), kMaxReturns));
        Color c;
        for (int i = 0; i < n; ++i) {
            const auto& r = returns[i];
            const float t = by == ColourBy::Height  ? r.position.y / 2.2f// ground to head height
                            : by == ColourBy::Range ? r.distance / range
                                                    : r.intensity;
            c.setHSL(0.66f * (1.f - std::clamp(t, 0.f, 1.f)), 1.f, 0.55f);
            position.setXYZ(i, r.position.x, r.position.y, r.position.z);
            colour.setXYZ(i, c.r, c.g, c.b);
        }
        geometry.setDrawRange(0, n);
        position.needsUpdate();
        colour.needsUpdate();
    }

}// namespace

int main(int argc, char** argv) {

    const auto args = capture::parseArgs(argc, argv);
    const bool shot = args.out.has_value();
    bool returnsOnly = false;
    for (int i = 1; i < argc; ++i) returnsOnly |= std::string(argv[i]) == "--returns";

    Canvas canvas("Lidar", {{"aa", 4}, {"size", WindowSize{1280, 720}}, {"headless", shot}, {"vsync", !shot}});
    GLRenderer renderer(canvas);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.shadowMap().enabled = true;

    auto scene = Scene::create();
    RGBELoader hdr;
    std::shared_ptr<Texture> sky = hdr.load(data + "/textures/env/kloofendal_48d_partly_cloudy_puresky_2k.hdr");
    scene->environment = sky;
    const auto haze = FogExp2(0xc6d0da, 0.02f);// fades the far ground into the sky

    // The sun casts the shadows. Once the renderer has found the sun in the sky
    // (Renderer::envSunFound), it is turned to shine from there.
    auto sun = DirectionalLight::create(0xfff4e8, 2.5f);
    sun->position.set(10, 20, 8);
    sun->castShadow = true;
    sun->shadow->mapSize.set(2048, 2048);
    auto* shadowCamera = sun->shadow->camera->as<OrthographicCamera>();
    shadowCamera->left = shadowCamera->bottom = -12;
    shadowCamera->right = shadowCamera->top = 12;
    shadowCamera->updateProjectionMatrix();
    scene->add(sun);
    bool sunAligned = false;

    buildYard(*scene);
    addCar(*scene);
    const auto walker = addWalker(*scene);

    constexpr float sensorHeight = 1.6f;
    scene->add(makeTripod(sensorHeight));

    int sensorIndex = 0;
    auto lidar = sensors[sensorIndex].make();
    lidar->position.y = sensorHeight;
    scene->addRef(*lidar);

    auto cloud = makeCloud();
    scene->add(cloud);

    // Under the returns-only view: a dark backdrop and a 1 m grid, on layer 2.
    auto grid = GridHelper::create(80, 80, 0x3a4048, 0x1e2126);
    grid->layers.set(2);
    scene->add(grid);

    auto camera = PerspectiveCamera::create(50, canvas.aspect(), 0.1f, 1000);
    camera->position.copy(args.camPos.value_or(Vector3(-7.5f, 4.2f, 8.5f)));
    OrbitControls controls{*camera, canvas};
    controls.target.copy(args.camTarget.value_or(Vector3(0.8f, 0.7f, 0)));
    controls.update();

    std::vector<const char*> sensorNames;
    for (const auto& s : sensors) sensorNames.push_back(s.name);
    const char* colourNames[] = {"Height", "Range", "Intensity"};
    int colourBy = 0;
    std::vector<LidarReturn> returns;
    float scanMs = 0;

    RendererSettingsUi ui(canvas, renderer, [&] {
        if (ImGui::Combo("Sensor", &sensorIndex, sensorNames.data(), static_cast<int>(sensorNames.size()))) {
            auto next = sensors[sensorIndex].make();
            next->position.copy(lidar->position);
            next->rangeNoise = lidar->rangeNoise;
            scene->remove(*lidar);
            lidar = std::move(next);
            scene->addRef(*lidar);
        }
        ImGui::SliderFloat("Range noise (m)", &lidar->rangeNoise.stddev, 0.f, 0.1f);
        ImGui::Combo("Colour by", &colourBy, colourNames, 3);
        ImGui::Checkbox("Returns only", &returnsOnly);
        ImGui::Text("%zu returns, scan %.1f ms", returns.size(), scanMs);
    }, "Lidar");

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    Clock clock;
    float time = 0;
    int frame = 0;
    canvas.animate([&] {
        const float dt = shot ? 1.f / 60 : clock.getDelta();
        time += dt;
        if (walker) walker->update(time, dt);

        if (!sunAligned && renderer.envSunFound()) {
            sun->position.copy(renderer.envSunDirection()).multiplyScalar(30);
            sunAligned = true;
        }

        // One scan a frame, stamped with the sim time.
        lidar->setSimTime(time);
        const auto start = std::chrono::steady_clock::now();
        lidar->scan(renderer, *scene, returns);
        scanMs = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
        showReturns(*cloud, returns, static_cast<ColourBy>(colourBy), lidar->far());

        // The scene with the returns over it, or the returns alone over the grid.
        camera->layers.set(returnsOnly ? 1 : 0);
        camera->layers.enable(returnsOnly ? 2 : 1);
        if (returnsOnly) {
            scene->background = Color(0x0c0e11);
            scene->fog.reset();
        } else {
            scene->background = sky;
            scene->fog = haze;
        }

        renderer.render(*scene, *camera);
        ui.render();

        if (shot && ++frame == args.frames.value_or(120)) {
            renderer.writeFramebuffer(*args.out);
            canvas.close();
        }
    });
}
