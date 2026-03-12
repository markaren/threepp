
#include "threepp/canvas/Monitor.hpp"
#include "threepp/controls/TransformControls.hpp"
#include "threepp/objects/TextSprite.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

namespace {

    class TransformKeyListener: public KeyListener {
    public:
        explicit TransformKeyListener(TransformControls& controls)
            : controls_(controls) {}

        void onKeyPressed(KeyEvent evt) override;
        void onKeyReleased(KeyEvent evt) override;

    private:
        TransformControls& controls_;
    };

}// namespace

int main() {

    Canvas canvas("Transform controls");
    canvas.exitOnKeyEscape(false);

    GLRenderer renderer(canvas.size());
    renderer.shadowMap().enabled = true;
    renderer.shadowMap().type = ShadowMap::PFC;
    renderer.autoClear = false;

    PerspectiveCamera camera(60, canvas.aspect(), 0.1f, 1000.f);
    camera.position.set(0, 5, 5);

    Scene scene;
    scene.background = Color::black;

    scene.add(AmbientLight::create(0xaaaaaa));

    TextureLoader tl;
    auto tex = tl.load(std::string(DATA_FOLDER) + "/textures/crate.gif");

    auto material = MeshBasicMaterial::create();
    material->map = tex;
    auto object = Mesh::create(BoxGeometry::create(), material);
    object->position.y = .5f;
    scene.add(object);


    auto grid = GridHelper::create(10, 10);
    scene.add(grid);

    OrbitControls orbitControls(camera, canvas);

    TransformControls controls(camera, canvas);
    controls.attach(*object);
    scene.add(controls);

    LambdaEventListener changeListener([&](Event& event) {
        orbitControls.enabled = !std::any_cast<bool>(event.target);
    });

    controls.addEventListener("dragging-changed", changeListener);

    TransformKeyListener keyListener(controls);
    canvas.addKeyListener(keyListener);


    HUD hud(renderer);
    FontLoader fontLoader;
    TextSprite text(fontLoader.defaultFont(), 20.f * monitor::contentScale().first);
    text.setText("Press Q to toggle space, W/E/R to change mode, X/Y/Z to toggle axis, hold SHIFT for snapping");
    text.setColor(Color::white);
    hud.add(text)
            .setNormalizedPosition({0.f, 0.f})
            .setVerticalAlignment(HUD::VerticalAlignment::ABOVE)
            .setHorizontalAlignment(HUD::HorizontalAlignment::LEFT);


    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();

        renderer.setSize(size);
    });

    canvas.animate([&] {
        renderer.clear();
        renderer.render(scene, camera);
        hud.render();
    });
}


void TransformKeyListener::onKeyPressed(KeyEvent evt) {
    switch (evt.key) {
        case Key::Q: {
            controls_.setSpace(controls_.getSpace() == "local" ? "world" : "local");
            break;
        }
        case Key::W: {
            controls_.setMode("translate");
            break;
        }
        case Key::E: {
            controls_.setMode("rotate");
            break;
        }
        case Key::R: {
            controls_.setMode("scale");
            break;
        }
        case Key::X: {
            controls_.showX = !controls_.showX;
            break;
        }
        case Key::Y: {
            controls_.showY = !controls_.showY;
            break;
        }
        case Key::Z: {
            controls_.showZ = !controls_.showZ;
            break;
        }
        case Key::SPACE: {
            controls_.enabled = !controls_.enabled;
            break;
        }
        case Key::LEFT_SHIFT:
            controls_.setTranslationSnap(1.f);
            controls_.setRotationSnap(math::degToRad(15.f));
            controls_.setScaleSnap(0.25f);
        default:
            break;
    }
}

void TransformKeyListener::onKeyReleased(KeyEvent evt) {
    if (evt.key == Key::LEFT_SHIFT) {
        controls_.setTranslationSnap(std::nullopt);
        controls_.setRotationSnap(std::nullopt);
        controls_.setScaleSnap(std::nullopt);
    }
}
