
#include "threepp/objects/TextSprite.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas("TextSprite");
    GLRenderer renderer(canvas.size());

    Scene scene;
    PerspectiveCamera camera(75, canvas.aspect(), 0.1f, 1000);
    camera.position.set(0,5,5);

    FontLoader fontLoader;
    auto font = fontLoader.load(std::string(DATA_FOLDER) + "/fonts/typeface/gentilis_regular.typeface.json");
    auto node = TextSprite::create(*font);
    node->setColor(Color::red);
    node->setText("Hello world");
    node->position.set(0, 0, 0);

    scene.add(node);

    auto grid = GridHelper::create(20, 10);
    scene.add(grid);

    OrbitControls controls(camera, canvas);

    Clock clock;
    canvas.animate([&] {

        const auto dt = clock.getDelta();
        node->setText(std::to_string(dt));

        renderer.render(scene, camera);
    });

}