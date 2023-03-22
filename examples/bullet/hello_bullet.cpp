
#include "threepp/threepp.hpp"

#include "threepp/extras/bullet/BulletWrapper.hpp"

#include <sstream>

#include <iostream>

using namespace threepp;

namespace {

    auto createTennisBallMaterial(TextureLoader& tl) {
        auto m = MeshPhongMaterial::create();
        m->map = tl.load("data/textures/NewTennisBallColor.jpg");
        m->bumpMap = tl.load("data/textures/TennisBallBump.jpg");
        return m;
    }

    auto createBox(TextureLoader& tl) {
        const auto geometry = BoxGeometry::create();
        const auto material = MeshPhongMaterial::create();
        material->map = tl.load("data/textures/crate.gif");
        return Mesh::create(geometry, material);
    }

    auto createSphere(TextureLoader& tl) {
        const auto geometry = SphereGeometry::create(0.5, 32, 32);
        const auto material = MeshPhongMaterial::create();
        material->map = tl.load("data/textures/uv_grid_opengl.jpg");
        return Mesh::create(geometry, material);
    }

    auto createCylinder(TextureLoader& tl) {
        const auto geometry = CylinderGeometry::create(0.5, 0.5, 1, 16, 4);
        const auto material = MeshPhongMaterial::create();
        material->map = tl.load("data/textures/uv_grid_opengl.jpg");
        return Mesh::create(geometry, material);
    }

    auto createCone(TextureLoader& tl) {
        const auto geometry = ConeGeometry::create(0.5, 1);
        const auto material = MeshPhongMaterial::create();
        material->map = tl.load("data/textures/uv_grid_opengl.jpg");
        return Mesh::create(geometry, material);
    }

    auto createCapsule(TextureLoader& tl) {
        const auto geometry = CapsuleGeometry::create(0.5, 1);
        const auto material = MeshPhongMaterial::create();
        material->map = tl.load("data/textures/uv_grid_opengl.jpg");
        return Mesh::create(geometry, material);
    }

    auto createPlane(TextureLoader& tl) {
        const auto planeGeometry = PlaneGeometry::create(20, 20);
        planeGeometry->rotateX(math::DEG2RAD * -90);
        const auto planeMaterial = MeshPhongMaterial::create();
        planeMaterial->map = tl.load("data/textures/checker.png");
        return Mesh::create(planeGeometry, planeMaterial);
    }

}// namespace

int main() {

    std::cerr << "Warning: This Demo uses a depreacted API. See the others demos instead" << std::endl;

    Canvas canvas(Canvas::Parameters().antialiasing(4));

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.set(-10, 10, 10);

    OrbitControls controls{camera, canvas};

    GLRenderer renderer(canvas);
    renderer.setClearColor(Color::aliceblue);

    scene->add(HemisphereLight::create());

    TextureLoader tl;
    auto box = createBox(tl);
    auto sphere = createSphere(tl);
    auto cylinder = createCylinder(tl);
    auto cone = createCone(tl);
    auto capsule = createCapsule(tl);
    auto plane = createPlane(tl);

    box->position.set(0, 4, 0);
    sphere->position.set(0, 5, 0.5);
    cylinder->position.set(0, 5, -0.5);
    cylinder->rotateZ(math::DEG2RAD * 45);
    cone->position.set(0, 4, 0);
    capsule->position.set(0, 6, 1);
    capsule->rotateZ(math::DEG2RAD * -45);

    scene->add(box);
    scene->add(sphere);
    scene->add(cylinder);
    scene->add(cone);
    scene->add(capsule);
    scene->add(plane);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    BulletWrapper bullet(Vector3::Y() * -9.81f);

    bullet.addRigidbody(RbWrapper::create(box->geometry(), 2), *box);
    bullet.addRigidbody(RbWrapper::create(sphere->geometry(), 2), *sphere);
    bullet.addRigidbody(RbWrapper::create(cylinder->geometry(), 5), *cylinder);
    bullet.addRigidbody(RbWrapper::create(cone->geometry(), 5), *cone);
    bullet.addRigidbody(RbWrapper::create(capsule->geometry(), 3), *capsule);
    bullet.addRigidbody(RbWrapper::create(plane->geometry()), *plane);

    auto tennisBallMaterial = createTennisBallMaterial(tl);

    KeyAdapter keyListener(KeyAdapter::Mode::KEY_PRESSED | threepp::KeyAdapter::KEY_REPEAT, [&](KeyEvent evt) {
        if (evt.key == 32) {// space
            auto geom = SphereGeometry::create(0.1);
            auto mesh = Mesh::create(geom, tennisBallMaterial->clone());
            mesh->position.copy(camera->position);
            auto rb = RbWrapper::create(geom.get(), 2);
            Vector3 dir;
            camera->getWorldDirection(dir);
            rb->body->setLinearVelocity(convert(dir * 10));
            bullet.addRigidbody(rb, *mesh);
            scene->add(mesh);

            canvas.invokeLater([mesh] {
                mesh->removeFromParent();
            },
                               2);
        }
    });
    canvas.addKeyListener(&keyListener);

    renderer.enableTextRendering();
    auto& handle = renderer.textHandle();

    float t = 0;
    canvas.animate([&](float dt) {
        bullet.step(dt);

        renderer.render(scene, camera);
        t += dt;

        std::stringstream ss;
        ss << renderer.info();
        handle.setText(ss.str());
    });
}
