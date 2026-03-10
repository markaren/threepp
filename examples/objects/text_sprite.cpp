
#include "threepp/objects/TextSprite.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas("TextSprite");
    GLRenderer renderer(canvas.size());

    Scene scene;
    PerspectiveCamera camera(75, canvas.aspect(), 0.1f, 1000);
    camera.position.set(0,5,5);

    auto node = TextSprite::create(std::string(DATA_FOLDER) + "/fonts/truetype/Roboto-Regular.ttf");
    node->setText("Hello world");
    node->position.set(0, 2, 0);
    scene.add(node);

    auto grid = GridHelper::create(20, 10);
    scene.add(grid);

    OrbitControls controls(camera, canvas);

    canvas.animate([&] {
        renderer.render(scene, camera);
    });

}