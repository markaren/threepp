
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas("Morphtargets");
    GLRenderer renderer(canvas.size());

    auto camera = PerspectiveCamera::create();

    auto scene = Scene::create();
    scene->add(camera);

    auto geometry = BoxGeometry::create(2, 2, 2, 32, 32, 32);
    auto& positionAttribute = geometry->morphAttributes["position"];


    canvas.animate([&] {

    });
}