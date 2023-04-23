
#include "AgxPhysics.hpp"


using namespace threepp;

auto createRotator() {

    auto body = new agx::RigidBody(new agxCollide::Geometry(new agxCollide::Cylinder(0.5, 0.05)));
    return body;
}

int main() {

    Canvas canvas;
    GLRenderer renderer{canvas};

    auto scene = Scene::create();

    auto rotator1 = createRotator();
    rotator1->setPosition({-10, 0, 0});
    auto rotator2 = createRotator();
    rotator2->setPosition({10, 0, 0});

    //    double radius = 0.1;
    //    auto cable = new agxCable::Cable(radius, 3);


    canvas.animate([&] {

    });
}