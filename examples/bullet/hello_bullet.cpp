
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

    const auto planeGeometry = PlaneGeometry::create(20, 20);
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

    BulletWrapper bullet(Vector3::Y * -9.81f);

    bullet.addRigidbody(RbWrapper::create(boxGeometry, 1), box);
    bullet.addRigidbody(RbWrapper::create(sphereGeometry, 2), sphere);
    bullet.addRigidbody(RbWrapper::create(cylinderGeometry, 5), cylinder);
    bullet.addRigidbody(RbWrapper::create(planeGeometry), plane);

    KeyAdapter keyListener(KeyAdapter::Mode::KEY_PRESSED | threepp::KeyAdapter::KEY_REPEAT, [&](KeyEvent evt){
       if (evt.key == 32) { // space
           auto geom = SphereGeometry::create(0.1);
           auto mat = MeshBasicMaterial::create();
           auto mesh = Mesh::create(geom, mat);
           mesh->position.copy(camera->position);
           auto rb = RbWrapper::create(geom, 10);
           Vector3 dir;
           camera->getWorldDirection(dir);
           rb->body->setLinearVelocity(convert(dir * 10));
           bullet.addRigidbody(rb, mesh);
           scene->add(mesh);
       }
    });
    canvas.addKeyListener(&keyListener);

    canvas.animate([&](float dt) {
        bullet.step(dt);

        renderer.render(scene, camera);
    });

}
