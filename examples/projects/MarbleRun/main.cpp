// A rolling-ball sculpture, and a tour of what the OpenGL renderer can do with
// nothing but its own geometry: no model, texture or HDR file is loaded.
//
// A brass screw lifts the marbles up the middle, two tracks wind them back down
// to a lacquered bowl, and the bowl swirls them in to the foot of the screw
// again (Sculpture.hpp). The steel marbles mirror the room, the glass ones
// refract what is behind them, and the glowing ones carry a light of their own
// and set off the hoops they roll through, each of which rings a note of its
// own (Chimes.hpp). One of them carries a camera.
//
// On the renderer's side: image-based light from a room that is itself built
// of boxes, soft shadows, clearcoat and transmission, instancing, a second view
// rendered to a texture, and a post chain of ambient occlusion, depth of field
// and bloom on linear HDR, tone-mapped once at the end.
//
//   mouse        orbit and zoom
//   O            the camera circles by itself, on / off
//   V            ride along: the view from the marble's camera
//   C            the camera moves to the next glowing marble
//   Space        slow motion
//   M            the chimes on / off (they start off)
//   1 2 3 4 5    ambient occlusion, bloom, depth of field, shadows, the floor's polish on / off
//
// Headless (dev):
//   MarbleRun --shot out.png [--time T] [--cam X Y Z] [--target X Y Z] [--size WxH]
//             [--ride] [--plain] [--circle] [--chimes] [--aperture A] [--marbles N]
//   MarbleRun --sequence <dir> [--fps F] ...
//   (--plain turns the post chain off, --ride starts in the view from the marble,
//   --circle keeps the camera circling from a --cam position, --chimes writes
//   the notes although a capture is silent)

#include "threepp/threepp.hpp"

#include "threepp/canvas/Monitor.hpp"
#include "threepp/extras/environments/RoomEnvironment.hpp"
#include "threepp/loaders/FontLoader.hpp"
#include "threepp/objects/Reflector.hpp"
#include "threepp/objects/TextSprite.hpp"
#include "threepp/postprocessing/postprocessing.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/renderers/gl/PMREMGenerator.hpp"

#include "Chimes.hpp"
#include "Sculpture.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

using namespace threepp;

namespace {

    constexpr int feedWidth = 360;
    constexpr int feedHeight = 225;

    // The floor, and over it the polish of the floor: a mirror image of the
    // scene, added to the lit floor by how squarely one looks along it, and
    // fading out with distance so that it needs no edge.
    std::shared_ptr<Reflector> setupStudio(Scene& scene, WindowSize size) {

        auto material = MeshStandardMaterial::create();
        material->color = Color(0x1c2025);
        material->roughness = 0.82f;

        auto floor = Mesh::create(PlaneGeometry::create(40, 40), material);
        floor->rotation.x = -math::PI / 2;
        floor->receiveShadow = true;
        scene.add(floor);

        Reflector::Options options;
        options.color = Color(1.f, 1.f, 1.f);
        options.textureWidth = static_cast<unsigned int>(size.width());
        options.textureHeight = static_cast<unsigned int>(size.height());
        options.shader = Shader{
                UniformMap{
                        {"color", Uniform()},
                        {"tDiffuse", Uniform()},
                        {"textureMatrix", Uniform()}},

                R"(
                uniform mat4 textureMatrix;
                varying vec4 vUv;
                varying vec3 vWorld;

                void main() {
                    vUv = textureMatrix * vec4( position, 1.0 );
                    vec4 world = modelMatrix * vec4( position, 1.0 );
                    vWorld = world.xyz;
                    gl_Position = projectionMatrix * viewMatrix * world;
                })",

                R"(
                uniform vec3 color;
                uniform sampler2D tDiffuse;
                varying vec4 vUv;
                varying vec3 vWorld;

                void main() {
                    vec3 reflection = texture2DProj( tDiffuse, vUv ).rgb;
                    vec3 toEye = normalize( cameraPosition - vWorld );
                    float fresnel = 0.16 + 0.84 * pow( 1.0 - max( toEye.y, 0.0 ), 5.0 );
                    float fade = 1.0 - smoothstep( 1.4, 2.6, length( vWorld.xz ) );
                    gl_FragColor = vec4( reflection * color * fresnel * fade, 1.0 );
                    #include <tonemapping_fragment>
                    #include <encodings_fragment>
                })"};

        auto polish = Reflector::create(CircleGeometry::create(2.7f, 96), options);
        polish->rotation.x = -math::PI / 2;
        polish->position.y = 0.002f;
        polish->material()->transparent = true;
        polish->material()->blending = Blending::Additive;
        polish->material()->depthWrite = false;
        scene.add(polish);

        return polish;
    }

    std::shared_ptr<DirectionalLight> setupLights(Scene& scene) {

        scene.add(HemisphereLight::create(0xdfe9ff, 0x2a2d33, 0.25f));

        auto key = DirectionalLight::create(0xfff1df, 1.4f);
        key->position.set(1.1f, 2.3f, 1.4f);
        key->castShadow = true;
        key->shadow->mapSize.set(4096, 4096);
        key->shadow->bias = -0.0003f;
        key->shadow->radius = 3.f;
        auto shadowCamera = key->shadow->camera->as<OrthographicCamera>();
        shadowCamera->left = shadowCamera->bottom = -1.f;
        shadowCamera->right = shadowCamera->top = 1.f;
        shadowCamera->nearPlane = 0.5f;
        shadowCamera->farPlane = 6.f;
        shadowCamera->updateProjectionMatrix();
        scene.add(key);

        // from behind, to pick the rails out of the dark
        auto rim = DirectionalLight::create(0x9fc4ff, 0.6f);
        rim->position.set(-1.6f, 1.3f, -1.5f);
        scene.add(rim);

        return key;
    }

    // The overlay: what is switched on, the keys, and the view from the marble.
    // It is a scene of its own, drawn over the finished frame, so that the
    // post chain leaves it alone.
    struct Hud {
        Scene scene;
        std::shared_ptr<TextSprite> values, riding;

        explicit Hud(const std::shared_ptr<Texture>& feed) {
            const float s = monitor::contentScale().first;

            FontLoader fontLoader;
            const auto font = fontLoader.defaultFont();

            const auto text = [&](const std::string& string, float size, const Color& color, float anchorX, float anchorY, float x, float y) {
                auto sprite = TextSprite::create(font, size * s, TextBox::Line);
                sprite->setText(string);
                sprite->setColor(color);
                sprite->setHorizontalAlignment(anchorX > 0.5f ? TextSprite::HorizontalAlignment::Right : TextSprite::HorizontalAlignment::Left);
                sprite->setVerticalAlignment(anchorY > 0.5f ? TextSprite::VerticalAlignment::Below : TextSprite::VerticalAlignment::Above);
                sprite->screenSpace = true;
                sprite->screenAnchor.set(anchorX, anchorY);
                sprite->position.set(x * s, y * s, 0);
                scene.add(sprite);

                return sprite;
            };

            // A rectangle, placed by the corner that its anchor names.
            const auto quad = [&](const std::shared_ptr<SpriteMaterial>& material, float anchorX, float anchorY, float x, float y, float width, float height) {
                auto sprite = Sprite::create(material);
                sprite->center.set(anchorX, anchorY);
                sprite->screenSpace = true;
                sprite->screenAnchor.set(anchorX, anchorY);
                sprite->position.set(x * s, y * s, 0);
                sprite->scale.set(width * s, height * s, 1);
                scene.add(sprite);

                return sprite;
            };

            auto panel = SpriteMaterial::create();
            panel->color = Color(0x0b0d10);
            panel->transparent = true;
            panel->opacity = 0.72f;
            panel->toneMapped = false;

            const Color dim(0x9fb0c0);
            quad(panel, 0, 1, 12, -12, 262, 216);
            text("marbles\nlaps\nframe\n\n1  ambient occlusion\n2  bloom\n3  depth of field\n4  shadows\n5  polished floor\nM  chimes", 17, dim, 0, 1, 24, -22);
            values = text(" ", 17, Color::white, 0, 1, 206, -22);

            quad(panel, 1, 0, -12, 12, 348, 62);
            text("mouse orbit   O circle   Space slow motion\n"
                 "V ride along   C next marble   M chimes\n"
                 "1 2 3 4 5 effects on / off",
                 14, dim, 1, 0, -22, 22);

            quad(panel, 0, 0, 12, 12, feedWidth + 10, feedHeight + 34);
            riding = text("marble camera", 14, dim, 0, 0, 19, feedHeight + 24);

            auto feedMaterial = SpriteMaterial::create();
            feedMaterial->map = feed;
            feedMaterial->toneMapped = false;
            quad(feedMaterial, 0, 0, 17, 17, feedWidth, feedHeight);
        }
    };

}// namespace

int main(int argc, char** argv) {

    std::string shotPath, sequenceDir;
    bool rideView = false, plain = false, circle = true, forceCircle = false, forceChimes = false;
    float shotTime = 2;
    int sequenceFps = 30;
    int width = 1280, height = 720;
    int marbleCount = 96;
    float aperture = 0.006f;
    Vector3 cameraPosition(1.4f, 0.6f, 1.65f), target(0, 0.33f, 0);
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shot" && i + 1 < argc) shotPath = argv[++i];
        else if (a == "--sequence" && i + 1 < argc) sequenceDir = argv[++i];
        else if (a == "--fps" && i + 1 < argc) sequenceFps = std::atoi(argv[++i]);
        else if (a == "--time" && i + 1 < argc) shotTime = std::stof(argv[++i]);
        else if (a == "--marbles" && i + 1 < argc) marbleCount = std::atoi(argv[++i]);
        else if (a == "--aperture" && i + 1 < argc) aperture = std::stof(argv[++i]);
        else if (a == "--ride") rideView = true;
        else if (a == "--plain") plain = true;
        else if (a == "--circle") forceCircle = true;
        else if (a == "--chimes") forceChimes = true;
        else if (a == "--size" && i + 1 < argc) std::sscanf(argv[++i], "%dx%d", &width, &height);
        else if ((a == "--cam" || a == "--target") && i + 3 < argc) {
            (a == "--cam" ? cameraPosition : target).set(std::stof(argv[i + 1]), std::stof(argv[i + 2]), std::stof(argv[i + 3]));
            circle = false;
            i += 3;
        }
    }
    circle = circle || forceCircle;
    const bool headless = !shotPath.empty() || !sequenceDir.empty();
    if (!sequenceDir.empty()) std::filesystem::create_directories(sequenceDir);

    Canvas canvas(Canvas::Parameters()
                          .title("Marble run")
                          .size(width, height)
                          .antialiasing(4)
                          .headless(headless));
    GLRenderer renderer(canvas);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.shadowMap().enabled = true;
    renderer.shadowMap().type = ShadowMap::PFC;

    const Color backdrop(0x14171b);
    Scene scene;
    scene.background = backdrop;
    scene.fog = Fog(backdrop, 3, 11);
    {
        PMREMGenerator pmrem(renderer);
        RoomEnvironment room;
        scene.environment = pmrem.fromScene(room, 0.04f);
        room.dispose();
    }
    scene.environmentIntensity = 0.45f;

    const auto polish = setupStudio(scene, canvas.size());
    const auto key = setupLights(scene);

    marblerun::Sculpture sculpture(marbleCount);
    scene.addRef(sculpture);

    PerspectiveCamera camera(32, canvas.aspect(), 0.02f, 40);// (orbitFov, below)
    camera.position.copy(cameraPosition);

    OrbitControls controls(camera, canvas);
    controls.target.copy(target);
    controls.enableKeys = false;
    controls.minDistance = 0.3f;
    controls.maxDistance = 6.f;
    controls.maxPolarAngle = math::PI / 2 - 0.03f;
    controls.autoRotateSpeed = 0.5f;
    controls.update();

    // The camera that rides with a marble, and what it sees.
    PerspectiveCamera rideCamera(68, static_cast<float>(feedWidth) / feedHeight, 0.004f, 40);
    RenderTarget::Options feedOptions;
    feedOptions.encoding = ColorSpace::sRGB;
    feedOptions.samples = 4;
    RenderTarget feed(feedWidth, feedHeight, feedOptions);
    feed.displayTarget = true;

    Hud hud(feed.texture);

    EffectComposer::Options composerOptions;
    composerOptions.samples = 4;
    EffectComposer composer(renderer, composerOptions);
    composer.addPass(std::make_shared<RenderPass>(scene, camera));

    const auto size = canvas.size();

    GTAOParameters aoParameters;
    aoParameters.radius = 0.05f;
    aoParameters.distanceExponent = 1.f;
    aoParameters.thickness = 1.f;
    aoParameters.scale = 1.f;
    aoParameters.samples = 16;
    aoParameters.distanceFallOff = 1.f;
    aoParameters.screenSpaceRadius = false;
    PoissonDenoiseParameters denoiseParameters;
    denoiseParameters.lumaPhi = 10.f;
    denoiseParameters.depthPhi = 2.f;
    denoiseParameters.normalPhi = 3.f;
    denoiseParameters.radius = 4.f;
    denoiseParameters.radiusExponent = 1.f;
    denoiseParameters.rings = 2.f;
    denoiseParameters.samples = 16;
    auto ao = std::make_shared<GTAOPass>(scene, camera, size.width(), size.height(), aoParameters, denoiseParameters);
    composer.addPass(ao);

    auto dof = std::make_shared<BokehPass>(scene, camera, 1.8f, aperture, 0.004f);
    composer.addPass(dof);

    auto bloom = std::make_shared<UnrealBloomPass>(Vector2(static_cast<float>(size.width()), static_cast<float>(size.height())), 0.14f, 0.1f, 1.3f);
    composer.addPass(bloom);

    if (plain) {
        ao->enabled = dof->enabled = bloom->enabled = false;
        polish->visible = false;
    }

    // A frame draws the scene several times over: for the marble's camera,
    // for the image, and again for the depths and normals that the post chain
    // works from. The shadows are the same in all of them, and the floor's
    // mirror image is only wanted in the image, so both are made once a frame.
    renderer.shadowMapAutoUpdate = false;
    bool mirrorDue = false;
    polish->onBeforeRender = RenderCallback([&mirrorDue, mirror = *polish->onBeforeRender](void* r, Object3D* scene, Camera* camera, BufferGeometry* geometry, Material* material, std::optional<GeometryGroup> group) {
        if (!mirrorDue) return;
        mirrorDue = false;
        mirror(r, scene, camera, geometry, material, group);
    });

    // A capture is silent (with --chimes it still writes the notes, to check them).
    std::unique_ptr<marblerun::Chimes> chimes;
    if (!headless || forceChimes) {
        try {
            // heard from the hoops, by the camera: riding a marble, one hears them come and go
            chimes = std::make_unique<marblerun::Chimes>(sculpture.gatePositions(), camera, sculpture);
            chimes->muted = true;// until asked for
        } catch (const std::exception& e) {
            std::cerr << "no sound: " << e.what() << std::endl;
        }
    }
    if (chimes && !headless) {
        sculpture.onGate = [&](int gate, float speed) { chimes->ring(gate, speed); };
    }

    size_t riding = 0;// which of the glowing marbles carries the camera
    bool slow = false;
    canvas.onKeyPressed([&](KeyEvent evt) {
        switch (evt.key) {
            case Key::NUM_1: ao->enabled = !ao->enabled; break;
            case Key::NUM_2: bloom->enabled = !bloom->enabled; break;
            case Key::NUM_3: dof->enabled = !dof->enabled; break;
            case Key::NUM_4: key->castShadow = !key->castShadow; break;
            case Key::NUM_5: polish->visible = !polish->visible; break;
            case Key::O: circle = !circle; break;
            case Key::V: rideView = !rideView; break;
            case Key::C: riding = (riding + 1) % sculpture.glowing().size(); break;
            case Key::SPACE: slow = !slow; break;
            case Key::M:
                if (chimes) chimes->muted = !chimes->muted;
                break;
            default: break;
        }
    });

    canvas.onWindowResize([&](WindowSize newSize) {
        camera.aspect = newSize.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(newSize);
        composer.setSize(newSize.width(), newSize.height());
    });

    // Where the riding camera is and what it looks at, eased so that it
    // does not jump as its marble steps on and off the lift.
    Vector3 rideEye, rideAim;
    bool rideSettled = false;
    const auto ride = [&](float dt) {
        const auto& m = sculpture.marbles()[sculpture.glowing()[riding]];
        const Vector3 up(0, 1, 0);

        Vector3 eye, aim;
        if (m.onLift) {
            // from outside the lift, looking up along it
            const Vector3 out = Vector3(m.position.x, 0, m.position.z).normalize();
            eye.copy(m.position).addScaledVector(out, 0.2f).addScaledVector(m.heading, 0.08f).addScaledVector(up, -0.03f);
            aim.copy(m.position).addScaledVector(up, 0.08f);
        } else {
            eye.copy(m.position).addScaledVector(m.heading, -0.17f).addScaledVector(up, 0.075f);
            aim.copy(m.position).addScaledVector(m.heading, 0.22f);
        }

        const float ease = rideSettled ? 1 - std::exp(-6.f * dt) : 1.f;
        rideEye.lerp(eye, ease);
        rideAim.lerp(aim, ease);
        rideSettled = true;
    };

    // the orbiting view, kept for when the ride is over
    constexpr float orbitFov = 32;
    Vector3 orbitPosition = camera.position;
    bool wasRiding = false;

    Clock clock;
    float time = 0, sinceHud = 1, frameTime = 1.f / 60;
    int frame = 0, written = 0;
    canvas.animate([&] {
        const float elapsed = std::min(clock.getDelta(), 0.05f);
        const float dt = (headless ? 1.f / 60 : elapsed) * (slow ? 0.2f : 1.f);
        time += dt;
        frameTime += 0.05f * (elapsed - frameTime);

        sculpture.update(dt);
        ride(dt);

        rideCamera.position.copy(rideEye);
        rideCamera.lookAt(rideAim);

        const float fov = rideView ? rideCamera.fov : orbitFov;
        if (camera.fov != fov) {
            camera.fov = fov;
            camera.updateProjectionMatrix();
        }
        if (rideView) {
            camera.position.copy(rideEye);
            camera.lookAt(rideAim);
            dof->focus = 0.3f;
        } else {
            if (wasRiding) camera.position.copy(orbitPosition);
            controls.autoRotate = circle;
            controls.update();
            orbitPosition.copy(camera.position);
            dof->focus = camera.position.distanceTo(controls.target);
        }
        wasRiding = rideView;

        renderer.shadowMap().needsUpdate = true;

        // the marble's camera sends every other frame
        if (frame % 2 == 0) {
            const bool polished = polish->visible;
            polish->visible = false;
            renderer.setRenderTarget(&feed);
            renderer.render(scene, rideCamera);
            renderer.setRenderTarget(nullptr);
            polish->visible = polished;
        }

        mirrorDue = true;
        composer.render(dt);

        sinceHud += elapsed;
        if (sinceHud >= 0.25f || headless) {
            sinceHud = 0;

            const auto onOff = [](bool on) { return on ? "on" : "off"; };
            char buffer[160];
            std::snprintf(buffer, sizeof(buffer), "%d\n%d\n%.1f ms\n\n%s\n%s\n%s\n%s\n%s\n%s", static_cast<int>(sculpture.marbles().size()),
                          sculpture.laps(), frameTime * 1000, onOff(ao->enabled), onOff(bloom->enabled), onOff(dof->enabled), onOff(key->castShadow), onOff(polish->visible), onOff(chimes && !chimes->muted));
            if (buffer != hud.values->getText()) hud.values->setText(buffer);
        }

        renderer.autoClear = false;
        renderer.clearDepth();
        renderer.render(hud.scene, camera);
        renderer.autoClear = true;

        if (!sequenceDir.empty() && frame % std::max(1, 60 / sequenceFps) == 0) {
            char name[32];
            std::snprintf(name, sizeof(name), "frame_%04d.png", written++);
            renderer.writeFramebuffer(std::filesystem::path(sequenceDir) / name);
        }
        ++frame;
        if (headless && time >= shotTime) {
            if (!shotPath.empty()) {
                renderer.writeFramebuffer(shotPath);
                std::cout << "wrote " << shotPath << std::endl;
            }
            std::exit(0);
        }
    });
}
