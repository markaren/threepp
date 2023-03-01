
#include "threepp/threepp.hpp"

#include "threepp/extras/physics/BulletPhysics.hpp"

#include <sstream>

using namespace threepp;

namespace {

    auto createTennisBallMaterial(TextureLoader& tl) {
        auto m = MeshPhongMaterial::create();
        m->map = tl.loadTexture("data/textures/NewTennisBallColor.jpg");
        m->bumpMap = tl.loadTexture("data/textures/TennisBallBump.jpg");
        return m;
    }

    auto createSpheres(TextureLoader& tl, unsigned int count) {
        const auto geometry = SphereGeometry::create(0.25, 32, 32);
        const auto material = MeshPhongMaterial::create();
        material->map = tl.loadTexture("data/textures/uv_grid_opengl.jpg");
        return InstancedMesh::create(geometry, material, count);
    }

    auto createPlane(TextureLoader& tl) {
        const auto planeGeometry = PlaneGeometry::create(100, 100);
        planeGeometry->rotateX(math::DEG2RAD * -90);
        const auto planeMaterial = MeshPhongMaterial::create();
        planeMaterial->map = tl.loadTexture("data/textures/checker.png");
        return Mesh::create(planeGeometry, planeMaterial);
    }

}// namespace

int main() {

    Canvas canvas(Canvas::Parameters().antialiasing(4));
    GLRenderer renderer(canvas);
    renderer.setClearColor(Color::aliceblue);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.set(-20, 10, 20);

    OrbitControls controls{camera, canvas};

    scene->add(HemisphereLight::create());

    TextureLoader tl;
    auto spheres = createSpheres(tl, 1500);
    spheres->instanceMatrix->setUsage(DynamicDrawUsage);
    auto plane = createPlane(tl);

    Matrix4 matrix;
    for (unsigned i = 0; i < spheres->count; i++) {

        matrix.setPosition( math::randomInRange(-10.f, 10.f), math::randomInRange(5.f, 20.f) , math::randomInRange(-10.f, 10.f));
        spheres->setMatrixAt(i, matrix);
    }

    scene->add(spheres);
    scene->add(plane);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    BulletPhysics bullet;

    bullet.addMesh(*spheres, 2);
    bullet.addMesh(*plane);

    auto tennisBallMaterial = createTennisBallMaterial(tl);

    KeyAdapter keyListener(KeyAdapter::Mode::KEY_PRESSED | threepp::KeyAdapter::KEY_REPEAT, [&](KeyEvent evt) {
        if (evt.key == 32) {// space
            auto geom = SphereGeometry::create(0.1);
            auto mesh = Mesh::create(geom, tennisBallMaterial->clone());
            mesh->position.copy(camera->position);
            mesh->rotation.set(math::randomInRange(0.f, math::TWO_PI), math::randomInRange(0.f, math::TWO_PI), math::randomInRange(0.f, math::TWO_PI));
            Vector3 dir;
            camera->getWorldDirection(dir);
            bullet.addMesh(*mesh, 1);
            bullet.get(*mesh)->body->setLinearVelocity(tobtVector(dir * 10));
            scene->add(mesh);

            canvas.invokeLater([mesh] { mesh->removeFromParent(); }, 2);
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
