
#include "threepp/objects/TextSprite.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas("TextNode");
    auto renderer = createRenderer(canvas);

    Scene scene;
    PerspectiveCamera camera(75, canvas.aspect(), 0.1f, 1000);
    camera.position.z = 5;

    auto node = TextSprite::create("C:/Windows/Fonts/Arial.ttf");
    node->setText("Hello world");
    node->position.set(0, 10, 0);
    scene.add(node);

    auto grid = GridHelper::create(20, 10);
    scene.add(grid);

    OrbitControls controls(camera, canvas);

    canvas.animate([&] {
        renderer->render(scene, camera);
    });

}