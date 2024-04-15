
#include "PxEngine.hpp"

#include "threepp/threepp.hpp"

#include <iostream>

using namespace threepp;

namespace {

    struct KeyController: KeyListener {

        explicit KeyController(std::vector<physx::PxRevoluteJoint*> joints)
            : joints(std::move(joints)) {

            for (auto j : this->joints) {
                j->setRevoluteJointFlag(physx::PxRevoluteJointFlag::eDRIVE_ENABLED, true);
            }
        }

        void onKeyPressed(KeyEvent evt) override;
        void onKeyReleased(KeyEvent evt) override;

    private:
        float speed = 2;
        std::vector<physx::PxRevoluteJoint*> joints;
    };

    void KeyController::onKeyPressed(KeyEvent evt) {
        if (evt.key == Key::NUM_1) {
            joints[0]->setDriveVelocity(-speed);
        } else if (evt.key == Key::NUM_2) {
            joints[0]->setDriveVelocity(speed);
        } else if (evt.key == Key::NUM_3) {
            joints[1]->setDriveVelocity(-speed);
        } else if (evt.key == Key::NUM_4) {
            joints[1]->setDriveVelocity(speed);
        } else if (evt.key == Key::NUM_5) {
            joints[2]->setDriveVelocity(-speed);
        } else if (evt.key == Key::NUM_6) {
            joints[2]->setDriveVelocity(speed);
        }
    }

    void KeyController::onKeyReleased(KeyEvent evt) {
        switch (evt.key) {
            case Key::NUM_1:
            case Key::NUM_2: {
                joints[0]->setDriveVelocity(0);
            } break;
            case Key::NUM_3:
            case Key::NUM_4: {
                joints[1]->setDriveVelocity(0);
            } break;
            case Key::NUM_5:
            case Key::NUM_6: {
                joints[2]->setDriveVelocity(0);
            } break;
            default:
                break;
        }
    }

    void setupInstancedMesh(InstancedMesh& mesh) {

        Matrix4 matrix;
        size_t index = 0;
        int amount = static_cast<int>(std::cbrt(mesh.count()));
        float offset = static_cast<float>(amount - 1) / 2;
        for (int x = 0; x < amount; x++) {
            for (int y = 0; y < amount; y++) {
                for (int z = 0; z < amount; z++) {
                    matrix.setPosition(offset - float(x), offset - float(y), offset - float(z));
                    mesh.setMatrixAt(index, matrix);
                    ++index;
                }
            }
        }
    }

    auto spawnObject() {

        auto geometry = SphereGeometry::create(0.1);
        auto material = MeshPhongMaterial::create();

        auto mesh = Mesh::create(geometry, material);
        mesh->castShadow = true;

        RigidBodyInfo info;
        info.setMass(10).setMaterialProperties(0.5, 0.9);
        mesh->userData["rigidbodyInfo"] = info;

        return mesh;
    }

}// namespace

int main() {

    PxEngine engine;

    Canvas canvas("PhysX", {{"aa", 6}});
    GLRenderer renderer(canvas.size());
    renderer.shadowMap().enabled = true;

    Scene scene;
    scene.background = Color::aliceblue;

    PerspectiveCamera camera(60, canvas.aspect());
    camera.position.set(0, 2.5, -8);

    TextureLoader tl;
    auto tex = tl.load("data/textures/checker.png");
    auto defaultMaterial = MeshPhongMaterial::create({{"color", Color::darkblue}});

    auto box1 = Mesh::create(BoxGeometry::create(0.5, 1, 0.5), defaultMaterial);
    box1->position.y = 0.5;
    box1->castShadow = true;
    scene.add(box1);

    auto box2 = Mesh::create(BoxGeometry::create(0.25, 2, 0.25), defaultMaterial);
    box2->position.y = 1.5;
    box2->castShadow = true;
    box1->add(box2);

    auto box3 = Mesh::create(CylinderGeometry::create(0.15, 0.15, 0.5), defaultMaterial);
    box3->position.y = 1.25;
    box3->castShadow = true;
    box2->add(box3);

    STLLoader loader;
    auto geometry = loader.load("data/models/stl/pr2_head_pan.stl");
    geometry->scale(5, 5, 5);
    auto sphereMaterial = MeshPhongMaterial::create({{"color", Color::red}});
    auto mesh = Mesh::create(geometry, sphereMaterial);
    mesh->castShadow = true;
    mesh->position.y = 10;
    scene.add(mesh);

    int groundSize = 60;
    auto groundMaterial = MeshPhongMaterial::create({{"color", Color::gray}});
    auto ground = Mesh::create(BoxGeometry::create(groundSize, 0.1, groundSize), groundMaterial);
    ground->position.y = -0.05;
    ground->receiveShadow = true;
    scene.add(ground);

    auto instanceMaterial = MeshPhongMaterial::create();
    instanceMaterial->color = Color::gray;
    auto instanced = InstancedMesh::create(SphereGeometry::create(0.25), instanceMaterial, static_cast<int>(std::pow(3, 3)));
    setupInstancedMesh(*instanced);
    instanced->position.y = 50;
    instanced->castShadow = true;
    scene.add(instanced);

    auto light1 = AmbientLight::create(Color::white, 0.3f);
    scene.add(light1);

    auto light2 = DirectionalLight::create();
    light2->position.set(100, 100, -100);
    light2->castShadow = true;
    auto shadowCamera = light2->shadow->camera->as<OrthographicCamera>();
    shadowCamera->left = shadowCamera->bottom = -10;
    shadowCamera->right = shadowCamera->top = 10;
    scene.add(light2);

    auto box1Body = RigidBodyInfo{};
    box1Body.addJoint()
            .setType(JointInfo::Type::HINGE)
            .setAnchor({0, -0.5, 0})
            .setAxis({0, 1, 0})
            .setLimits({math::degToRad(-120), math::degToRad(120)});
    box1->userData["rigidbodyInfo"] = box1Body;

    auto box2Body = RigidBodyInfo{};
    box2Body.addJoint()
            .setType(JointInfo::Type::HINGE)
            .setAnchor({0, -1, 0})
            .setAxis({0, 0, 1})
            .setConnectedBody(*box1)
            .setLimits({math::degToRad(0), math::degToRad(120)});
    box2->userData["rigidbodyInfo"] = box2Body;

    auto box3Body = RigidBodyInfo{};
    box3Body.addJoint()
            .setType(JointInfo::Type::HINGE)
            .setAnchor({0, -0.25, 0})
            .setAxis({0, 1, 0})
            .setConnectedBody(*box2)
            .setLimits({math::degToRad(-90), math::degToRad(90)});
    box3->userData["rigidbodyInfo"] = box3Body;

    auto groundBody = RigidBodyInfo{RigidBodyInfo::Type::STATIC}
                              .addCollider(BoxCollider(0.5, 2, groundSize / 2), Matrix4().setPosition(groundSize / 2, 2, 0))
                              .addCollider(BoxCollider(0.5, 2, groundSize / 2), Matrix4().setPosition(-groundSize / 2, 2, 0))
                              .addCollider(BoxCollider(0.5, 2, groundSize / 2), Matrix4().makeRotationY(math::PI / 2).setPosition(0, 2, groundSize / 2))
                              .addCollider(BoxCollider(0.5, 2, groundSize / 2), Matrix4().makeRotationY(math::PI / 2).setPosition(0, 2, -groundSize / 2));
    ground->userData["rigidbodyInfo"] = groundBody;

    auto meshBody = RigidBodyInfo{}.setMass(20).setMaterialProperties(1, 0);
    mesh->userData["rigidbodyInfo"] = meshBody;

    auto instancedBody = RigidBodyInfo();
    instanced->userData["rigidbodyInfo"] = instancedBody;

    engine.setup(scene);
    scene.add(engine);

    auto* j1 = engine.getJoint<physx::PxRevoluteJoint>(*box1);
    auto* j2 = engine.getJoint<physx::PxRevoluteJoint>(*box2);
    auto* j3 = engine.getJoint<physx::PxRevoluteJoint>(*box3);

    KeyController keyListener({j1, j2, j3});
    canvas.addKeyListener(keyListener);

    OrbitControls controls(camera, canvas);

    bool run = false;
    KeyAdapter adapter(KeyAdapter::Mode::KEY_PRESSED | KeyAdapter::KEY_REPEAT, [&](KeyEvent evt) {
        run = true;

        if (evt.key == Key::SPACE) {
            auto obj = spawnObject();
            obj->position = camera.position;
            scene.add(obj);
            engine.setup(*obj);
            auto rb = engine.getBody(*obj)->is<physx::PxRigidDynamic>();
            Vector3 world;
            camera.getWorldDirection(world);
            rb->addForce(toPxVector3(world * 10000));

            renderer.invokeLater([&, obj] {
                scene.remove(*obj);
            },
                               2);// remove after 2 seconds
        } else if (evt.key == Key::D) {
            engine.debugVisualisation = !engine.debugVisualisation;
        }
    });
    canvas.addKeyListener(adapter);

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(size);
    });

    std::cout << "Press any key to start.\n"
                 "\nPress 'd' to toggle debug visualization."
                 "\nPress 'space' to spawn spheres."
              << std::endl;

    Clock clock;
    canvas.animate([&] {
        auto dt = clock.getDelta();

        renderer.render(scene, camera);

        if (run) {
            engine.step(dt);
        }
    });
}
