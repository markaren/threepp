
#include "threepp/threepp.hpp"

#include "threepp/extras/bullet/BulletWrapper.hpp"

#include <sstream>

using namespace threepp;

auto createTennisBallMaterial(TextureLoader& tl) {
    auto m = MeshPhongMaterial::create();
    m->map = tl.loadTexture("data/textures/NewTennisBallColor.jpg");
    m->bumpMap = tl.loadTexture("data/textures/TennisBallBump.jpg");
    return m;
}

int main() {

    Canvas canvas;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.set(-5, 5, 5);

    OrbitControls controls{camera, canvas};

    GLRenderer renderer(canvas);
    renderer.setClearColor(Color::aliceblue);

    scene->add(HemisphereLight::create());

    TextureLoader tl;
    const auto boxGeometry = BoxGeometry::create();
    const auto boxMaterial = MeshPhongMaterial::create();
    boxMaterial->map = tl.loadTexture("data/textures/crate.gif");
    auto box = Mesh::create(boxGeometry, boxMaterial);
    box->position.setY(6);
    scene->add(box);

    const auto sphereGeometry = SphereGeometry::create(0.5);
    const auto sphereMaterial = MeshPhongMaterial::create();
    sphereMaterial->map = tl.loadTexture("data/textures/uv_grid_opengl.jpg");
    auto sphere = Mesh::create(sphereGeometry, sphereMaterial);
    sphere->position.set(0, 5, 0.5);
    scene->add(sphere);

    const auto cylinderGeometry = CylinderGeometry::create(0.5, 0.5);
    const auto cylinderMaterial = MeshPhongMaterial::create();
    cylinderMaterial->map = tl.loadTexture("data/textures/uv_grid_opengl.jpg");
    auto cylinder = Mesh::create(cylinderGeometry, cylinderMaterial);
    cylinder->position.set(0, 5, -0.5);
    cylinder->rotateZ(math::DEG2RAD*45);
    scene->add(cylinder);

    const auto planeGeometry = PlaneGeometry::create(20, 20);
    planeGeometry->rotateX(math::DEG2RAD*-90);
    const auto planeMaterial = MeshPhongMaterial::create();
    planeMaterial->map = TextureLoader().loadTexture("data/textures/checker.png");
    auto plane = Mesh::create(planeGeometry, planeMaterial);
    scene->add(plane);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    BulletWrapper bullet(Vector3::Y * -9.81f);

    bullet.addRigidbody(RbWrapper::create(boxGeometry, 1), *box);
    bullet.addRigidbody(RbWrapper::create(sphereGeometry, 2), *sphere);
    bullet.addRigidbody(RbWrapper::create(cylinderGeometry, 5), *cylinder);
    bullet.addRigidbody(RbWrapper::create(planeGeometry), *plane);

    auto tennisBallMaterial = createTennisBallMaterial(tl);

    KeyAdapter keyListener(KeyAdapter::Mode::KEY_PRESSED | threepp::KeyAdapter::KEY_REPEAT, [&](KeyEvent evt){
       if (evt.key == 32) { // space
           auto geom = SphereGeometry::create(0.1);
           auto mesh = Mesh::create(geom, tennisBallMaterial->clone());
           mesh->position.copy(camera->position);
           auto rb = RbWrapper::create(geom, 10);
           Vector3 dir;
           camera->getWorldDirection(dir);
           rb->body->setLinearVelocity(convert(dir * 10));
           bullet.addRigidbody(rb, *mesh);
           scene->add(mesh);

           canvas.invokeLater([mesh]{
               mesh->removeFromParent();
           }, 2);
       }
    });
    canvas.addKeyListener(&keyListener);

    renderer.enableTextRendering();
    auto& handle = renderer.textHandle();

    float t = 0;
    canvas.animate([&](float dt) {
        bullet.step(dt);

        renderer.render(scene, camera);
        t+=dt;

        std::stringstream ss;
        ss << renderer.info();
        handle.setText(ss.str());
    });

}
