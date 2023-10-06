
#include "threepp/extras/physics/BulletPhysics.hpp"
#include "threepp/threepp.hpp"

#include "threepp/lights/LightShadow.hpp"

#include <sstream>

using namespace threepp;

namespace {

    auto createTennisBallMaterial(TextureLoader& tl) {
        auto m = MeshPhongMaterial::create();
        m->map = tl.load("data/textures/NewTennisBallColor.jpg");
        m->bumpMap = tl.load("data/textures/TennisBallBump.jpg");
        return m;
    }

    auto createSpheres(TextureLoader& tl, unsigned int count) {
        const auto geometry = SphereGeometry::create(0.25f);
        const auto material = MeshPhongMaterial::create();
        material->map = tl.load("data/textures/uv_grid_opengl.jpg");
        return InstancedMesh::create(geometry, material, count);
    }

    auto createCapsules(TextureLoader& tl, unsigned int count) {
        const auto geometry = CapsuleGeometry::create(0.25f, 0.5f);
        const auto material = MeshPhongMaterial::create();
        material->map = tl.load("data/textures/uv_grid_opengl.jpg");
        return InstancedMesh::create(geometry, material, count);
    }

    auto createBoxes(TextureLoader& tl, unsigned int count) {
        const auto geometry = BoxGeometry::create(0.5f, 0.5f, 0.5f);
        const auto material = MeshPhongMaterial::create();
        material->map = tl.load("data/textures/uv_grid_opengl.jpg");
        return InstancedMesh::create(geometry, material, count);
    }

    auto createPlane(TextureLoader& tl) {
        const auto planeGeometry = PlaneGeometry::create(100, 100);
        planeGeometry->rotateX(math::DEG2RAD * -90);
        planeGeometry->translate(0, 0.1f, 0);
        const auto planeMaterial = MeshPhongMaterial::create();
        planeMaterial->color = Color::gray;
        return Mesh::create(planeGeometry, planeMaterial);
    }

    inline Vector3 randomPosition() {
        return {math::randFloat(-5.f, 5.f), math::randFloat(5.f, 15.f), math::randFloat(-5.f, 5.f)};
    }

    void initPositions(InstancedMesh& m) {
        Matrix4 matrix;
        for (unsigned i = 0; i < m.count; i++) {

            matrix.setPosition(randomPosition());
            m.setMatrixAt(i, matrix);
        }
    }

    auto createShadowLight() {
        auto shadowLight = DirectionalLight::create();
        shadowLight->position.set(100, 100, 100);
        shadowLight->castShadow = true;
        shadowLight->shadow->radius = 1;
        shadowLight->shadow->camera->as<OrthographicCamera>()->top = 100;
        shadowLight->shadow->camera->as<OrthographicCamera>()->bottom = -100;
        shadowLight->shadow->camera->as<OrthographicCamera>()->left = 100;
        shadowLight->shadow->camera->as<OrthographicCamera>()->right = -100;
        shadowLight->shadow->mapSize.multiplyScalar(10);
        return shadowLight;
    }

}// namespace

int main() {

    Canvas canvas("Instanced physics", {{"aa", 4}});
    GLRenderer renderer(canvas.size());
    renderer.shadowMap().enabled = true;
    renderer.shadowMap().type = ShadowMap::PFCSoft;
    renderer.setClearColor(Color::aliceblue);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.aspect(), 0.1f, 1000);
    camera->position.set(-20, 10, 20);

    OrbitControls controls{*camera, canvas};

    scene->add(AmbientLight::create(0xffffff, 0.1f));
    scene->add(createShadowLight());

    TextureLoader tl;
    unsigned int count = 250;
    auto spheres = createSpheres(tl, count);
    spheres->instanceMatrix->setUsage(DrawUsage::Dynamic);
    spheres->castShadow = true;
    auto capsules = createCapsules(tl, count);
    capsules->instanceMatrix->setUsage(DrawUsage::Dynamic);
    capsules->castShadow = true;
    auto boxes = createBoxes(tl, count);
    boxes->instanceMatrix->setUsage(DrawUsage::Dynamic);
    boxes->castShadow = true;
    auto plane = createPlane(tl);
    plane->receiveShadow = true;

    initPositions(*spheres);
    initPositions(*capsules);
    initPositions(*boxes);

    scene->add(spheres);
    scene->add(capsules);
    scene->add(boxes);
    scene->add(plane);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    BulletPhysics bullet;

    bullet.addMesh(*spheres, 2);
    bullet.addMesh(*capsules, 2);
    bullet.addMesh(*boxes, 2);
    bullet.addMesh(*plane);

    auto tennisBallGeom = SphereGeometry::create(0.1);
    auto tennisBallMaterial = createTennisBallMaterial(tl);

    KeyAdapter keyListener(KeyAdapter::Mode::KEY_PRESSED | threepp::KeyAdapter::KEY_REPEAT, [&](KeyEvent evt) {
        if (evt.key == Key::SPACE) {

            auto mesh = Mesh::create(tennisBallGeom, tennisBallMaterial);
            mesh->castShadow = true;
            mesh->position.copy(camera->position);
            mesh->rotation.set(math::randFloat() * math::TWO_PI, math::randFloat() * math::TWO_PI, math::randFloat() * math::TWO_PI);
            Vector3 dir;
            camera->getWorldDirection(dir);
            bullet.addMesh(*mesh, 1);
            bullet.get(*mesh)->body->setLinearVelocity(tobtVector(dir * 10));
            scene->add(mesh);

            canvas.invokeLater([mesh] { mesh->removeFromParent(); }, 2);
        } else if (evt.key == Key::R) {

            for (unsigned i = 0; i < count; i++) {

                bullet.setInstancedMeshPosition(*spheres, randomPosition(), i);
                bullet.setInstancedMeshPosition(*capsules, randomPosition(), i);
                bullet.setInstancedMeshPosition(*boxes, randomPosition(), i);
            }
        }
    });
    canvas.addKeyListener(&keyListener);

    TextRenderer textRenderer;
    auto& handle = textRenderer.createHandle();

    Clock clock;
    canvas.animate([&]() {
        float dt = clock.getDelta();
        bullet.step(dt);

        renderer.render(*scene, *camera);
        renderer.resetState();

        std::stringstream ss;
        ss << renderer.info();
        handle.setText(ss.str());

        textRenderer.render();
    });
}
