#include "renderer_factory.hpp"

#include "threepp/geometries/TorusKnotGeometry.hpp"
#include "threepp/input/PeripheralsEventSource.hpp"
#include "threepp/threepp.hpp"

#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    // Feeds the controls a scripted gesture instead of a real mouse, for --shot.
    // onMouse*Event is protected on PeripheralsEventSource, so a subclass is the
    // way in - the same trick the unit test uses.
    class ScriptedSource: public PeripheralsEventSource {

    public:
        explicit ScriptedSource(WindowSize size): size_(size) {}

        [[nodiscard]] WindowSize size() const override {

            return size_;
        }

        void press(const Vector2& pos) {

            onMousePressedEvent(0, pos, MouseAction::PRESS);
        }

        void move(const Vector2& pos) {

            onMouseMoveEvent(pos);
        }

        void release(const Vector2& pos) {

            onMousePressedEvent(0, pos, MouseAction::RELEASE);
        }

    private:
        WindowSize size_;
    };

    struct Gesture {
        Vector2 from;
        Vector2 to;
    };

}// namespace

int main(int argc, char** argv) {

    // --shot <prefix>: replay a fixed gesture and write <prefix>_1..3.png
    std::string shotPrefix;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--shot" && i + 1 < argc) shotPrefix = argv[++i];
    }

    Canvas canvas("TrackballControls", {{"aa", 4}});
    auto renderer = createRenderer(canvas);

    Scene scene;
    scene.background = Color(0x1b1b22);

    PerspectiveCamera camera(50, canvas.aspect(), 0.1f, 100);
    camera.position.set(0, 2.5f, 7);

    scene.add(AmbientLight::create(0x404050, 1.f));

    auto keyLight = DirectionalLight::create(0xffffff, 2.5f);
    keyLight->position.set(4, 6, 5);
    scene.add(keyLight);

    auto fillLight = DirectionalLight::create(0x8899ff, 1.2f);
    fillLight->position.set(-5, -4, -4);
    scene.add(fillLight);

    // an object with no meaningful "up" - exactly the case trackball is for
    auto knot = Mesh::create(
            TorusKnotGeometry::create(1.5f, 0.45f, 200, 32),
            MeshPhongMaterial::create(
                    MeshPhongMaterial::Params{}
                            .color(0xc8a24a)
                            .specular(0x777777)
                            .shininess(60.f)));
    scene.add(knot);

    // world axes, so the roll that trackball allows is actually visible
    scene.add(AxesHelper::create(3));

    if (!shotPrefix.empty()) {

        const auto size = canvas.size();
        const auto w = static_cast<float>(size.width());
        const auto h = static_cast<float>(size.height());

        ScriptedSource source(size);
        TrackballControls controls(camera, source);
        controls.staticMoving = true;

        // a square loop on the trackball, then two drags straight over the pole
        const std::vector<Gesture> gestures{
                {{w * 0.25f, h * 0.70f}, {w * 0.75f, h * 0.70f}},
                {{w * 0.75f, h * 0.70f}, {w * 0.75f, h * 0.30f}},
                {{w * 0.75f, h * 0.30f}, {w * 0.25f, h * 0.30f}},
                {{w * 0.25f, h * 0.30f}, {w * 0.25f, h * 0.70f}},
                {{w * 0.50f, h * 0.85f}, {w * 0.50f, h * 0.15f}},
                {{w * 0.50f, h * 0.85f}, {w * 0.50f, h * 0.15f}},
        };

        constexpr int STEPS = 24;
        constexpr int PER_GESTURE = STEPS + 1;// one press frame, then the moves

        int frame = 0;

        canvas.animate([&] {
            const int elapsed = frame - 1;// frame 0 is the untouched view

            if (elapsed >= 0) {

                const auto index = static_cast<std::size_t>(elapsed / PER_GESTURE);
                const int step = elapsed % PER_GESTURE;

                if (index < gestures.size()) {

                    const auto& gesture = gestures[index];

                    if (step == 0) {

                        source.press(gesture.from);

                    } else {

                        const auto t = static_cast<float>(step) / STEPS;
                        source.move({gesture.from.x + (gesture.to.x - gesture.from.x) * t,
                                     gesture.from.y + (gesture.to.y - gesture.from.y) * t});

                        if (step == STEPS) source.release(gesture.to);
                    }
                }
            }

            controls.update();
            renderer->render(scene, camera);

            const auto write = [&](const std::string& name) {
                renderer->writeFramebuffer(shotPrefix + name);
                std::cout << "wrote " << shotPrefix + name << std::endl;
            };

            if (frame == 0) {
                write("_1_start.png");
            } else if (elapsed == 4 * PER_GESTURE - 1) {
                write("_2_rolled.png");
            } else if (elapsed == 6 * PER_GESTURE - 1) {
                write("_3_pole.png");
                canvas.close();
            }

            ++frame;
        });

        return 0;
    }

    std::optional<TrackballControls> trackball;
    std::optional<OrbitControls> orbit;

    bool useTrackball = true;

    auto applyMode = [&] {
        // only one set of controls listens at a time
        trackball.reset();
        orbit.reset();

        if (useTrackball) {

            auto& controls = trackball.emplace(camera, canvas);
            controls.minDistance = 2.5f;
            controls.maxDistance = 40.f;
            controls.dynamicDampingFactor = 0.15f;

        } else {

            auto& controls = orbit.emplace(camera, canvas);
            controls.minDistance = 2.5f;
            controls.maxDistance = 40.f;
            controls.enableDamping = true;
        }

        std::cout << (useTrackball ? "TrackballControls" : "OrbitControls") << std::endl;
    };

    applyMode();

    KeyAdapter keyAdapter(KeyAdapter::Mode::KEY_PRESSED, [&](KeyEvent evt) {
        if (evt.key == Key::SPACE) {

            useTrackball = !useTrackball;
            applyMode();

        } else if (evt.key == Key::R) {

            // camera.up is what trackball rolls, so it has to be put back too
            camera.up.set(0, 1, 0);
            camera.position.set(0, 2.5f, 7);
            applyMode();
        }
    });
    canvas.addKeyListener(keyAdapter);

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();

        renderer->setSize(size);
    });

    std::cout << "Left mouse: rotate, right: pan, middle/wheel: zoom" << std::endl;
    std::cout << "Press 'space' to switch between trackball and orbit, 'r' to reset the view" << std::endl;
    std::cout << "Try tumbling straight over the top: trackball keeps going and lets the horizon roll," << std::endl;
    std::cout << "orbit stops dead at the pole and keeps the y-axis upright." << std::endl;

    canvas.animate([&] {
        if (trackball) {
            trackball->update();
        } else {
            orbit->update();
        }

        renderer->render(scene, camera);
    });
}
