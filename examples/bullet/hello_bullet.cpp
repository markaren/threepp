
#include "threepp/threepp.hpp"

#include "threepp/extras/bullet/BulletWrapper.hpp"

using namespace threepp;

int main() {

    Canvas canvas;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.set(-5, 5, 5);

    OrbitControls controls{camera, canvas};

    GLRenderer renderer(canvas);
    renderer.setClearColor(Color::aliceblue);

    const auto boxGeometry = BoxGeometry::create();
    const auto boxMaterial = MeshBasicMaterial::create();
    boxMaterial->color = Color::blue;
    auto box = Mesh::create(boxGeometry, boxMaterial);
    box->position.setY(6);
    scene->add(box);

    const auto sphereGeometry = SphereGeometry::create(0.5);
    const auto sphereMaterial = MeshBasicMaterial::create();
    sphereMaterial->color = Color::gray;
    sphereMaterial->wireframe = true;
    auto sphere = Mesh::create(sphereGeometry, sphereMaterial);
    sphere->position.set(0, 5, 0.5);
    scene->add(sphere);

    const auto cylinderGeometry = CylinderGeometry::create(0.5, 0.5);
    const auto cylinderMaterial = MeshBasicMaterial::create();
    cylinderMaterial->color = Color::gray;
    cylinderMaterial->wireframe = true;
    auto cylinder = Mesh::create(cylinderGeometry, cylinderMaterial);
    cylinder->position.set(0, 5, -0.5);
    cylinder->rotateZ(math::DEG2RAD*45);
    scene->add(cylinder);

    const auto planeGeometry = PlaneGeometry::create(10, 10);
    planeGeometry->rotateX(math::DEG2RAD*-90);
    const auto planeMaterial = MeshBasicMaterial::create();
    planeMaterial->color = Color::red;
    auto plane = Mesh::create(planeGeometry, planeMaterial);
    scene->add(plane);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    BulletWrapper bullet;
    bullet.setGravity(Vector3::Y * -9.81f);
    auto boxRb = RbWrapper(fromGeometry(boxGeometry), 1, *box->matrixWorld);
    bullet.addRigidbody(boxRb);

//    engine.registerMesh(box, 1);
//    engine.registerMesh(sphere, 2);
//    engine.registerMesh(cylinder, 1);
//    engine.registerMesh(plane, 0);

    canvas.animate([&](float dt) {
        bullet.step(dt);

        renderer.render(scene, camera);
    });

}
