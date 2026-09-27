// A GL showpiece: everything the OpenGL renderer does well, in one moving frame.
//
// A car turns on a neon turntable inside a vortex of chrome shards. Coloured
// lights orbit it, glass floats around it, and a ring of monoliths chases light
// round the stage. The camera flies a scripted path around it all.
//
// What is on screen:
//   - RoomEnvironment studio IBL (PMREM fromScene), dimmed by environmentIntensity
//   - soft spot-light shadows and a point-light cube shadow, both moving
//   - an InstancedMesh vortex (per-instance colour), glass with transmission,
//     iridescence and dispersion, a glTF car with its own PBR maps
//   - exponential fog, additive dust motes
//   - EffectComposer: MSAA RenderPass -> GTAO -> UnrealBloom -> Outline -> OutputPass -> FXAA
//
//   C        free camera (orbit) / flight path
//   Space    pause
//   G / B / O / F   GTAO / bloom / outline / FXAA on-off
//
// Headless (dev):
//   showpiece --shot out.png [--time T] [--size WxH]
//   showpiece --sequence <dir> [--frames N] [--fps F] [--time T0] [--size WxH]
//   either with --no-gtao / --no-bloom / --no-outline / --no-fxaa, --samples N (composer MSAA)

#include "threepp/extras/environments/RoomEnvironment.hpp"
#include "threepp/geometries/TorusKnotGeometry.hpp"
#include "threepp/input/KeyListener.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/postprocessing/postprocessing.hpp"
#include "threepp/renderers/gl/PMREMGenerator.hpp"
#include "threepp/threepp.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    struct Shard {
        float angle, radius, height, speed, phase, scale;
        Vector3 spinAxis;
        float spinRate;
    };

    struct Shot {
        Vector3 position;
        Vector3 target;
        float fov;
    };

    // The flight path: a slow orbit whose radius, height and aim all breathe at
    // unrelated rates, so it never quite repeats, plus an occasional crane up.
    Shot flightPath(float t) {
        const float angle = 0.12f * t + 0.6f * std::sin(0.05f * t);
        const float radius = 10.f + 2.f * std::sin(0.21f * t + 1.f);
        const float crane = std::pow(std::max(0.f, std::sin(0.07f * t - 0.5f)), 4.f);
        const float height = 0.6f + 2.4f * (0.5f + 0.5f * std::sin(0.17f * t)) + 4.f * crane;

        Shot s;
        s.position.set(radius * std::sin(angle), height, radius * std::cos(angle));
        s.target.set(0.8f * std::sin(0.3f * t), 0.9f + 0.3f * std::sin(0.23f * t), 0.8f * std::cos(0.27f * t));
        s.fov = 42.f + 8.f * std::sin(0.13f * t);
        return s;
    }

    std::shared_ptr<MeshStandardMaterial> emissive(const Color& color, float intensity) {
        auto m = MeshStandardMaterial::create();
        m->color = Color(0x050505);
        m->emissive = color;
        m->emissiveIntensity = intensity;
        m->roughness = 0.5f;
        return m;
    }

}// namespace


int main(int argc, char** argv) {

    std::string shotPath, sequenceDir;
    float startTime = 6.f, fps = 30.f;
    int frames = 300;
    int width = 1280, height = 720;
    bool noGtao = false, noBloom = false, noOutline = false;
    int samples = 4;
    bool noFxaa = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shot" && i + 1 < argc) shotPath = argv[++i];
        else if (a == "--sequence" && i + 1 < argc) sequenceDir = argv[++i];
        else if (a == "--time" && i + 1 < argc) startTime = std::stof(argv[++i]);
        else if (a == "--frames" && i + 1 < argc) frames = std::atoi(argv[++i]);
        else if (a == "--fps" && i + 1 < argc) fps = std::stof(argv[++i]);
        else if (a == "--size" && i + 1 < argc) std::sscanf(argv[++i], "%dx%d", &width, &height);
        else if (a == "--samples" && i + 1 < argc) samples = std::atoi(argv[++i]);
        else if (a == "--no-fxaa") noFxaa = true;
        else if (a == "--no-gtao") noGtao = true;
        else if (a == "--no-bloom") noBloom = true;
        else if (a == "--no-outline") noOutline = true;
    }
    const bool headless = !shotPath.empty() || !sequenceDir.empty();
    if (!sequenceDir.empty()) std::filesystem::create_directories(sequenceDir);

    Canvas canvas(Canvas::Parameters()
                          .title("threepp GL showpiece")
                          .size(width, height)
                          .antialiasing(8)
                          .headless(headless));
    GLRenderer renderer(canvas);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = 1.f;
    renderer.shadowMap().enabled = true;
    renderer.shadowMap().type = ShadowMap::PFC;

    const Color night(0x04050b);
    auto scene = Scene::create();
    scene->background = night;
    scene->fog = FogExp2(night, 0.032f);
    {
        PMREMGenerator pmrem(renderer);
        RoomEnvironment room;
        scene->environment = pmrem.fromScene(room, 0.04f);
    }
    scene->environmentIntensity = 0.45f;

    auto camera = PerspectiveCamera::create(45, canvas.aspect(), 0.1f, 200.f);
    OrbitControls controls{*camera, canvas};
    controls.enabled = false;
    bool freeCamera = false;

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u01(0.f, 1.f);

    // ---- stage ------------------------------------------------------------

    auto floorMaterial = MeshStandardMaterial::create();
    floorMaterial->color = Color(0x0b0c10);
    floorMaterial->roughness = 0.22f;
    floorMaterial->metalness = 0.6f;
    floorMaterial->envMapIntensity = 0.06f;// the studio room would turn it milky at grazing angles
    auto floor = Mesh::create(CircleGeometry::create(60, 96), floorMaterial);
    floor->rotation.x = -math::PI / 2;
    floor->receiveShadow = true;
    scene->add(floor);

    auto grid = GridHelper::create(80, 80, 0x1a3a50, 0x0c1a26);
    grid->position.y = 0.005f;
    scene->add(grid);

    // Turntable: dark brushed metal with a neon rim, and a second ring on the floor.
    auto turntable = Group::create();
    scene->add(turntable);
    {
        auto deckMaterial = MeshStandardMaterial::create();
        deckMaterial->color = Color(0x2a2c33);
        deckMaterial->metalness = 0.9f;
        deckMaterial->roughness = 0.28f;
        deckMaterial->envMapIntensity = 0.35f;
        auto deck = Mesh::create(CylinderGeometry::create(3.f, 3.1f, 0.22f, 96), deckMaterial);
        deck->position.y = 0.11f;
        deck->castShadow = true;
        deck->receiveShadow = true;
        turntable->add(deck);

        auto rim = Mesh::create(TorusGeometry::create(3.07f, 0.035f, 12, 160), emissive(Color(0x22ddff), 7.f));
        rim->rotation.x = math::PI / 2;
        rim->position.y = 0.22f;
        turntable->add(rim);
    }
    auto floorRingMaterial = emissive(Color(0xff2a8a), 4.f);
    auto floorRing = Mesh::create(TorusGeometry::create(6.8f, 0.03f, 8, 200), floorRingMaterial);
    floorRing->rotation.x = math::PI / 2;
    floorRing->position.y = 0.02f;
    scene->add(floorRing);

    // The car.
    std::shared_ptr<Object3D> car;
    {
        const std::string path = std::string(DATA_FOLDER) + "/models/gltf/porsche_911/scene.gltf";
        if (std::filesystem::exists(path)) {
            GLTFLoader loader;
            if (auto gltf = loader.load(path)) {
                auto model = gltf->scene;
                Box3 box;
                box.setFromObject(*model);
                Vector3 size, center;
                box.getSize(size);
                box.getCenter(center);
                const float scale = 4.4f / std::max(size.x, size.z);
                model->scale.setScalar(scale);
                model->position.set(-center.x * scale, 0.22f - box.min().y * scale, -center.z * scale);
                model->traverseType<Mesh>([](Mesh& m) {
                    m.castShadow = true;
                    m.receiveShadow = true;
                });
                auto holder = Group::create();
                holder->add(model);
                turntable->add(holder);
                car = holder;
            }
        }
        if (!car) {// no data folder: a chrome knot stands in
            auto chrome = MeshStandardMaterial::create();
            chrome->metalness = 1.f;
            chrome->roughness = 0.1f;
            auto knot = Mesh::create(TorusKnotGeometry::create(0.9f, 0.3f, 200, 32), chrome);
            knot->position.y = 1.6f;
            knot->castShadow = true;
            turntable->add(knot);
            car = knot;
        }
    }

    // Monoliths: a ring of glossy black slabs whose light strips chase round the stage.
    constexpr int monolithCount = 12;
    std::vector<std::shared_ptr<MeshStandardMaterial>> stripMaterials;
    {
        auto slabMaterial = MeshStandardMaterial::create();
        slabMaterial->color = Color(0x0a0a0e);
        slabMaterial->metalness = 0.7f;
        slabMaterial->roughness = 0.18f;
        auto slabGeometry = BoxGeometry::create(1.3f, 7.f, 0.5f);
        auto stripGeometry = BoxGeometry::create(0.08f, 6.4f, 0.08f);
        for (int i = 0; i < monolithCount; ++i) {
            const float a = static_cast<float>(i) / monolithCount * math::TWO_PI;
            auto slab = Mesh::create(slabGeometry, slabMaterial);
            slab->position.set(15.f * std::sin(a), 3.5f, 15.f * std::cos(a));
            slab->rotation.y = a;
            slab->castShadow = true;
            slab->receiveShadow = true;

            auto strip = emissive(i % 2 ? Color(0xff2a8a) : Color(0x22ddff), 1.f);
            stripMaterials.push_back(strip);
            for (float side: {-0.5f, 0.5f}) {
                auto s = Mesh::create(stripGeometry, strip);
                s->position.set(side * 1.3f, 0, 0.27f);
                slab->add(s);
            }
            scene->add(slab);
        }
    }

    // ---- the vortex -------------------------------------------------------

    constexpr int shardCount = 220;
    std::vector<Shard> shards(shardCount);
    auto shardMaterial = MeshStandardMaterial::create();
    shardMaterial->metalness = 1.f;
    shardMaterial->roughness = 0.2f;
    auto vortex = InstancedMesh::create(IcosahedronGeometry::create(1.f, 0), shardMaterial, shardCount);
    vortex->castShadow = true;
    vortex->frustumCulled = false;
    for (int i = 0; i < shardCount; ++i) {
        auto& s = shards[i];
        s.angle = u01(rng) * math::TWO_PI;
        s.radius = 3.8f + 1.6f * u01(rng);
        s.height = 0.5f + 3.6f * std::pow(u01(rng), 1.3f);
        s.speed = (0.25f + 0.35f * u01(rng)) * (5.f / s.radius);
        s.phase = u01(rng) * math::TWO_PI;
        s.scale = 0.035f + 0.07f * std::pow(u01(rng), 2.f);
        s.spinAxis.set(u01(rng) - 0.5f, u01(rng) - 0.5f, u01(rng) - 0.5f).normalize();
        s.spinRate = 0.5f + 2.5f * u01(rng);

        const float pick = u01(rng);
        Color c(0.5f, 0.52f, 0.56f);
        if (pick > 0.86f) c.setHSL(0.52f + 0.08f * u01(rng), 0.9f, 0.55f);// cyan
        else if (pick > 0.74f) c.setHSL(0.9f + 0.06f * u01(rng), 0.9f, 0.55f);// magenta
        else if (pick > 0.68f) c.setHSL(0.09f, 0.9f, 0.55f);// gold
        vortex->setColorAt(i, c);
    }
    scene->add(vortex);

    // ---- glass ------------------------------------------------------------

    std::vector<std::shared_ptr<Mesh>> glass;
    {
        auto knotGlass = MeshPhysicalMaterial::create();
        knotGlass->transmission = 1.f;
        knotGlass->roughness = 0.04f;
        knotGlass->thickness = 0.8f;
        knotGlass->setIor(1.5f);
        knotGlass->color = Color(0xbfe8ff);
        knotGlass->dispersion = 3.f;
        glass.push_back(Mesh::create(TorusKnotGeometry::create(0.45f, 0.15f, 220, 32), knotGlass));

        auto bubble = MeshPhysicalMaterial::create();
        bubble->transmission = 1.f;
        bubble->roughness = 0.f;
        bubble->thickness = 0.05f;
        bubble->iridescence = 1.f;
        bubble->iridescenceIOR = 1.3f;
        glass.push_back(Mesh::create(SphereGeometry::create(0.6f, 64, 32), bubble));

        auto gem = MeshPhysicalMaterial::create();
        gem->transmission = 1.f;
        gem->roughness = 0.02f;
        gem->thickness = 1.2f;
        gem->setIor(2.2f);
        gem->color = Color(0xffc0e0);
        gem->dispersion = 5.f;
        gem->flatShading = true;
        glass.push_back(Mesh::create(IcosahedronGeometry::create(0.55f, 0), gem));

        for (auto& g: glass) {
            g->castShadow = true;
            scene->add(g);
        }
    }

    // ---- lights -----------------------------------------------------------

    auto spot = SpotLight::create(0xfff2e0, 2.4f, 0, 0.36f, 0.6f, 0);
    spot->position.set(2.f, 15.f, 3.f);
    spot->castShadow = true;
    spot->shadow->mapSize.set(2048, 2048);
    spot->shadow->bias = -0.0004f;
    spot->shadow->radius = 3.f;
    scene->add(spot);

    struct Orb {
        std::shared_ptr<PointLight> light;
        std::shared_ptr<Mesh> bulb;
        float radius, height, speed, phase;
    };
    std::vector<Orb> orbs;
    {
        const std::array<std::tuple<int, float, float, float, float>, 3> spec{{
                {0x22ddff, 3.6f, 2.4f, 0.55f, 0.f},
                {0xff2a8a, 4.0f, 1.4f, -0.4f, 2.1f},
                {0xffa020, 3.2f, 3.3f, 0.7f, 4.2f},
        }};
        auto bulbGeometry = SphereGeometry::create(0.09f, 16, 8);
        for (size_t i = 0; i < spec.size(); ++i) {
            const auto [hex, r, h, speed, phase] = spec[i];
            Orb o{PointLight::create(hex, 6.f, 14.f, 2.f), Mesh::create(bulbGeometry, emissive(Color(hex), 30.f)), r, h, speed, phase};
            if (i == 0) {
                o.light->castShadow = true;
                o.light->shadow->mapSize.set(1024, 1024);
                o.light->shadow->bias = -0.002f;
                o.light->shadow->radius = 2.f;
            }
            o.light->add(o.bulb);
            scene->add(o.light);
            orbs.push_back(o);
        }
    }

    // ---- dust -------------------------------------------------------------

    constexpr int dustCount = 2500;
    std::vector<float> dustBase(dustCount * 3);
    for (int i = 0; i < dustCount; ++i) {
        const float a = u01(rng) * math::TWO_PI, r = 1.f + 13.f * std::sqrt(u01(rng));
        dustBase[i * 3 + 0] = r * std::sin(a);
        dustBase[i * 3 + 1] = 9.f * u01(rng);
        dustBase[i * 3 + 2] = r * std::cos(a);
    }
    auto dustGeometry = BufferGeometry::create();
    dustGeometry->setAttribute("position", FloatBufferAttribute::create(dustBase, 3));
    auto dustMaterial = PointsMaterial::create();
    dustMaterial->color = Color(0xffd9a8);
    dustMaterial->size = 0.035f;
    dustMaterial->transparent = true;
    dustMaterial->opacity = 0.7f;
    dustMaterial->blending = Blending::Additive;
    dustMaterial->depthWrite = false;
    auto dust = Points::create(dustGeometry, dustMaterial);
    dust->frustumCulled = false;
    scene->add(dust);

    // ---- post -------------------------------------------------------------

    EffectComposer::Options options;
    options.samples = samples;
    EffectComposer composer(renderer, options);
    composer.addPass(std::make_shared<RenderPass>(*scene, *camera));

    const auto size = canvas.size();
    GTAOParameters aoParameters;
    aoParameters.radius = 0.4f;
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
    gtao->blendIntensity = 0.9f;
    composer.addPass(gtao);

    const Vector2 resolution(static_cast<float>(size.width()), static_cast<float>(size.height()));
    auto bloom = std::make_shared<UnrealBloomPass>(resolution, 0.35f, 0.3f, 1.8f);
    composer.addPass(bloom);

    auto outline = std::make_shared<OutlinePass>(resolution, *scene, *camera);
    outline->edgeStrength = 4.f;
    outline->edgeThickness = 1.5f;
    outline->edgeGlow = 0.6f;
    outline->pulsePeriod = 2.f;
    composer.addPass(outline);

    // MSAA resolves in linear HDR, so edges against the emitters keep their
    // stairs; FXAA after the tone map catches those.
    composer.addPass(std::make_shared<OutputPass>());
    auto fxaa = std::make_shared<FXAAPass>();
    composer.addPass(fxaa);

    gtao->enabled = !noGtao;
    bloom->enabled = !noBloom;
    outline->enabled = !noOutline;
    fxaa->enabled = !noFxaa;

    // The outline hops between the car and the glass, like a scanner locking on.
    std::vector<Object3D*> outlineTargets{car.get()};
    for (auto& g: glass) outlineTargets.push_back(g.get());
    const std::array<Color, 4> outlineColors{Color(0x22ddff), Color(0xff2a8a), Color(0xffa020), Color(0xffffff)};

    // ---- animation --------------------------------------------------------

    Matrix4 m;
    Quaternion q;
    Vector3 p, sc;
    auto update = [&](float t) {
        turntable->rotation.y = 0.22f * t;

        for (int i = 0; i < shardCount; ++i) {
            const auto& s = shards[i];
            const float a = s.angle + s.speed * t;
            const float r = s.radius + 0.5f * std::sin(0.7f * t + s.height * 1.3f + s.phase);
            const float y = s.height + 0.35f * std::sin(3.f * a + s.phase) + 0.25f * std::sin(0.9f * t + s.phase);
            p.set(r * std::sin(a), y, r * std::cos(a));
            q.setFromAxisAngle(s.spinAxis, s.spinRate * t + s.phase);
            sc.set(s.scale, s.scale * 2.2f, s.scale);// long shards
            m.compose(p, q, sc);
            vortex->setMatrixAt(i, m);
        }
        vortex->instanceMatrix()->needsUpdate();

        for (size_t i = 0; i < glass.size(); ++i) {
            const float a = 0.3f * t + static_cast<float>(i) * math::TWO_PI / 3.f;
            glass[i]->position.set(3.4f * std::sin(a), 2.9f + 0.4f * std::sin(1.1f * t + static_cast<float>(i)), 3.4f * std::cos(a));
            glass[i]->rotation.set(0.7f * t + i, 0.5f * t, 0.3f * t * (i + 1.f));
        }

        for (auto& o: orbs) {
            const float a = o.speed * t + o.phase;
            o.light->position.set(o.radius * std::sin(a), o.height + 0.6f * std::sin(1.3f * t + o.phase), o.radius * std::cos(a));
        }

        for (int i = 0; i < monolithCount; ++i) {
            // A wave of light runs round the ring, twice a lap.
            const float phase = static_cast<float>(i) / monolithCount * math::TWO_PI * 2.f - 2.2f * t;
            stripMaterials[i]->emissiveIntensity = 0.4f + 9.f * std::pow(0.5f + 0.5f * std::sin(phase), 6.f);
        }
        floorRingMaterial->emissiveIntensity = 2.5f + 2.f * std::sin(1.7f * t);

        auto* pos = dustGeometry->getAttribute<float>("position");
        for (int i = 0; i < dustCount; ++i) {
            const float x = dustBase[i * 3], z = dustBase[i * 3 + 2];
            const float y = std::fmod(dustBase[i * 3 + 1] + 0.25f * t, 9.f);
            pos->setXYZ(i, x + 0.3f * std::sin(0.5f * t + z), y, z + 0.3f * std::cos(0.4f * t + x));
        }
        pos->needsUpdate();

        const int target = static_cast<int>(t / 4.f) % static_cast<int>(outlineTargets.size());
        outline->selectedObjects = {outlineTargets[target]};
        outline->visibleEdgeColor = outlineColors[target % outlineColors.size()];

        if (!freeCamera) {
            const auto shot = flightPath(t);
            camera->position.copy(shot.position);
            camera->fov = shot.fov;
            camera->updateProjectionMatrix();
            camera->lookAt(shot.target);
            controls.target.copy(shot.target);
        }
    };

    bool paused = false;
    canvas.onKeyPressed([&](KeyEvent evt) {
        switch (evt.key) {
            case Key::C:
                freeCamera = !freeCamera;
                controls.enabled = freeCamera;
                if (freeCamera) controls.update();
                break;
            case Key::SPACE: paused = !paused; break;
            case Key::G: gtao->enabled = !gtao->enabled; break;
            case Key::B: bloom->enabled = !bloom->enabled; break;
            case Key::O: outline->enabled = !outline->enabled; break;
            case Key::F: fxaa->enabled = !fxaa->enabled; break;
            default: break;
        }
    });

    canvas.onWindowResize([&](WindowSize s) {
        camera->aspect = s.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(s);
        composer.setSize(s.width(), s.height());
    });

    float t = startTime;
    int frame = 0;
    Clock clock;
    canvas.animate([&] {
        const float dt = headless ? 1.f / fps : clock.getDelta();

        if (!sequenceDir.empty()) {
            update(t);
            composer.render(dt);
            char name[64];
            std::snprintf(name, sizeof(name), "/frame_%05d.png", frame);
            renderer.writeFramebuffer(sequenceDir + name);
            t += dt;
            if (++frame >= frames) std::exit(0);
            return;
        }

        if (!paused) t += dt;
        update(t);
        composer.render(dt);

        if (!shotPath.empty() && ++frame >= 3) {
            renderer.writeFramebuffer(shotPath);
            std::cout << "wrote " << shotPath << std::endl;
            std::exit(0);
        }
    });
}
