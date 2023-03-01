
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

    auto createBox(TextureLoader& tl) {
        const auto geometry = BoxGeometry::create();
        const auto material = MeshPhongMaterial::create();
        material->map = tl.loadTexture("data/textures/crate.gif");
        return Mesh::create(geometry, material);
    }

    auto createSphere(TextureLoader& tl) {
        const auto geometry = SphereGeometry::create(0.5, 32, 32);
        const auto material = MeshPhongMaterial::create();
        material->map = tl.loadTexture("data/textures/uv_grid_opengl.jpg");
        return Mesh::create(geometry, material);
    }

    auto createCylinder(TextureLoader& tl) {
        const auto geometry = CylinderGeometry::create(0.5, 0.5, 1, 16, 4);
        const auto material = MeshPhongMaterial::create();
        material->map = tl.loadTexture("data/textures/uv_grid_opengl.jpg");
        return Mesh::create(geometry, material);
    }

    auto createCone(TextureLoader& tl) {
        const auto geometry = ConeGeometry::create(0.5, 1);
        const auto material = MeshPhongMaterial::create();
        material->map = tl.loadTexture("data/textures/uv_grid_opengl.jpg");
        return Mesh::create(geometry, material);
    }

    auto createCapsule(TextureLoader& tl) {
        const auto geometry = CapsuleGeometry::create(0.5, 1);
        const auto material = MeshPhongMaterial::create();
        material->map = tl.loadTexture("data/textures/uv_grid_opengl.jpg");
        return Mesh::create(geometry, material);
    }

    auto createPlane(TextureLoader& tl) {
        const auto geometry = PlaneGeometry::create(20, 20);
        geometry->rotateX(math::DEG2RAD * -90);
        const auto material = MeshPhongMaterial::create();
        material->map = tl.loadTexture("data/textures/checker.png");
        return Mesh::create(geometry, material);
    }

    auto createTriangleMesh(TextureLoader& tl) {
        STLLoader loader;
        const auto geometry = loader.load("data/models/stl/pr2_head_pan.stl");
        geometry->scale(5, 5, 5);
        const auto material = MeshPhongMaterial::create();
        material->color = Color::brown;
        material->flatShading = true;
        auto m = Mesh::create(geometry, material);
        auto wire = LineSegments::create(WireframeGeometry::create(*geometry));
        wire->material()->as<LineBasicMaterial>()->color = 0x000000;
        wire->material()->as<LineBasicMaterial>()->transparent = true;
        wire->material()->as<LineBasicMaterial>()->opacity = 0.5f;
        m->add(wire);
        return m;
    }

}// namespace

int main() {

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
    auto trimesh = createTriangleMesh(tl);

    box->position.set(0, 4, 0);
    sphere->position.set(0, 5, 0.5);
    cylinder->position.set(0, 5, -0.5);
    cylinder->rotateZ(math::DEG2RAD * 45);
    cone->position.set(0, 4, 0);
    capsule->position.set(0, 6, 1);
    capsule->rotateZ(math::DEG2RAD * -45);
    trimesh->position.set(1, 3, -2);

    scene->add(box);
    scene->add(sphere);
    scene->add(cylinder);
    scene->add(cone);
    scene->add(capsule);
    scene->add(plane);
    scene->add(trimesh);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    BulletPhysics bullet;

    bullet.addMesh(*box, 5);
    bullet.addMesh(*sphere, 3);
    bullet.addMesh(*cylinder, 4);
    bullet.addMesh(*capsule, 8);
    bullet.addMesh(*cone, 3);
    bullet.addMesh(*plane);
    bullet.addMesh(*trimesh, 10);

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
