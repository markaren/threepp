
#include "PxEngine.hpp"


using namespace threepp;

namespace {

    struct KeyController: KeyListener {

        explicit KeyController(std::vector<physx::PxRevoluteJoint*> joints)
            : joints(std::move(joints)) {

            for (auto j : this->joints) {
                j->setRevoluteJointFlag(physx::PxRevoluteJointFlag::eDRIVE_ENABLED, true);
            }
        }

        void onKeyPressed(KeyEvent evt) override {
            if (evt.key == Key::NUM_1) {
                joints[0]->setDriveVelocity(-5);
            } else if (evt.key == Key::NUM_2) {
                joints[0]->setDriveVelocity(5);
            } else if (evt.key == Key::NUM_3) {
                joints[1]->setDriveVelocity(-5);
            } else if (evt.key == Key::NUM_4) {
                joints[1]->setDriveVelocity(5);
            } else if (evt.key == Key::NUM_5) {
                joints[2]->setDriveVelocity(-5);
            } else if (evt.key == Key::NUM_6) {
                joints[2]->setDriveVelocity(5);
            }
        }
        void onKeyReleased(KeyEvent evt) override {
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

    private:
        float speed = 0.25;
        std::vector<physx::PxRevoluteJoint*> joints;
    };

    auto spawnObject() {

        auto geometry = SphereGeometry::create(0.1);
        auto material = MeshPhongMaterial::create();

        auto mesh = Mesh::create(geometry, material);

        RigidBodyInfo info;
        info.mass = 10.f;
        mesh->userData["rigidbodyInfo"] = info;

        return mesh;
    }

}// namespace

int main() {

    Canvas canvas("PhysX", {{"aa", 6}});
    GLRenderer renderer(canvas.size());

    Scene scene;
    scene.background = Color::aliceblue;

    PerspectiveCamera camera(60, canvas.aspect());
    camera.position.set(0, 2.5, -8);

    TextureLoader tl;
    auto tex = tl.load("data/textures/checker.png");
    auto defaultMaterial = MeshPhongMaterial::create({{"color", Color::darkblue}});

    auto box1 = Mesh::create(BoxGeometry::create(0.5, 1, 0.5), defaultMaterial);
    box1->position.y = 0.5;
    scene.add(box1);

    auto box2 = Mesh::create(BoxGeometry::create(0.25, 2, 0.25), defaultMaterial);
    box2->position.y = 1.5;
    box1->add(box2);

    auto box3 = Mesh::create(CylinderGeometry::create(0.15, 0.15, 0.5), defaultMaterial);
    box3->position.y = 1.25;
    box2->add(box3);

    auto sphereMaterial = MeshPhongMaterial::create({{"color", Color::red}});
    auto sphere = Mesh::create(SphereGeometry::create(0.5), sphereMaterial);
    sphere->position.y = 20;
    scene.add(sphere);

    auto groundMaterial = MeshPhongMaterial::create({{"color", Color::gray}});
    auto ground = Mesh::create(BoxGeometry::create(40, 0.1, 40), groundMaterial);
    ground->position.y = -0.05;
    scene.add(ground);

    auto light1 = HemisphereLight::create();
    light1->position.y = 5;
    scene.add(light1);

    PxEngine engine;

    auto box1Body = RigidBodyInfo{};
    box1Body.addJoint().setType(threepp::JointInfo::Type::HINGE).setAnchor({0, -0.5, 0}).setAxis({0, 1, 0}).setLimits({math::degToRad(-120), math::degToRad(120)});
    box1->userData["rigidbodyInfo"] = box1Body;

    auto box2Body = RigidBodyInfo{};
    box2Body.addJoint().setType(threepp::JointInfo::Type::HINGE).setAnchor({0, -1, 0}).setAxis({0, 0, 1}).setConnectedBody(*box1).setLimits({math::degToRad(0), math::degToRad(120)});
    box2->userData["rigidbodyInfo"] = box2Body;

    auto box3Body = RigidBodyInfo{};
    box3Body.addJoint().setType(threepp::JointInfo::Type::HINGE).setAnchor({0, -0.25, 0}).setAxis({0, 1, 0}).setConnectedBody(*box2).setLimits({math::degToRad(-90), math::degToRad(90)});
    box3->userData["rigidbodyInfo"] = box3Body;

    auto groundBody = RigidBodyInfo{RigidBodyInfo::Type::STATIC};
    ground->userData["rigidbodyInfo"] = groundBody;

    auto sphereBody = RigidBodyInfo{}.setMass(50);
    sphere->userData["rigidbodyInfo"] = sphereBody;

    engine.setup(scene);
    scene.add(engine);

    auto* j1 = engine.getJoint<physx::PxRevoluteJoint>(*box1);
    auto* j2 = engine.getJoint<physx::PxRevoluteJoint>(*box2);
    auto* j3 = engine.getJoint<physx::PxRevoluteJoint>(*box3);

    KeyController keyListener({j1, j2, j3});
    canvas.addKeyListener(keyListener);

    OrbitControls controls(camera, canvas);
    controls.enableKeys = true;

    bool run = false;
    KeyAdapter adapter(KeyAdapter::Mode::KEY_PRESSED | threepp::KeyAdapter::KEY_REPEAT, [&](KeyEvent evt) {
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

            canvas.invokeLater([&, obj]{
                scene.remove(*obj);
            }, 1);
        }
    });
    canvas.addKeyListener(adapter);

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(size);
    });

    Clock clock;
    canvas.animate([&] {
        auto dt = clock.getDelta();

        renderer.render(scene, camera);

        if (run) {
            engine.step(dt);
        }
    });
}
