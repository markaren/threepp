
#include "threepp/threepp.hpp"

using namespace threepp;

int main () {

    Canvas canvas;

    const auto scene = Scene::create();
    const auto camera = PerspectiveCamera::create( 75, canvas.getAspect(), 0.1f, 1000 );

    auto renderer = GLRenderer(canvas);
    renderer.setSize( canvas.getWidth(), canvas.getHeight() );

    const auto geometry = BoxGeometry::create();
    const auto material = MeshBasicMaterial::create( /*{ color: 0x00ff00 }*/ );
    const auto cube = Mesh::create( geometry, material );
    scene->add( cube );

    camera->position.z = 5;

    canvas.animate( [&](float dt) {

        cube->rotation.x(cube->rotation.x() + 0.01);
        cube->rotation.y(cube->rotation.y() + 0.01);

        //renderer.render( scene, camera );
    });

}
