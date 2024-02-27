
#include "PxEngine.hpp"


using namespace threepp;

namespace {

    struct KeyController: KeyListener {

        explicit KeyController(std::vector<physx::PxRevoluteJoint*> joints): joints(std::move(joints)) {}

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
        float speed = 5;
        std::vector<physx::PxRevoluteJoint*> joints;
    };

}// namespace

int main() {

    Canvas canvas("PhysX", {{"aa", 6}});
    GLRenderer renderer(canvas.size());

    Scene scene;
    scene.background = Color::aliceblue;

    PerspectiveCamera camera(60, canvas.aspect());
    camera.position.set(0, 2.5, 8);

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
    scene.add(engine);
    engine.registerMeshDynamic(*box1);
    engine.registerMeshDynamic(*box2);
    engine.registerMeshDynamic(*box3);
    engine.registerMeshDynamic(*sphere);
    engine.registerMeshStatic(*ground);

    auto joint1 = engine.createRevoluteJoint(*box1, {0, 0.05, 0}, {0, 1, 0});
    auto joint2 = engine.createRevoluteJoint(*box1, *box2, {0, 0.5, 0}, {1, 0, 0});
    auto joint3 = engine.createRevoluteJoint(*box2, *box3, {0, 0.25, 0}, {0, 1, 0});

    joint1->setRevoluteJointFlag(physx::PxRevoluteJointFlag::eDRIVE_ENABLED, true);
    joint1->setRevoluteJointFlag(physx::PxRevoluteJointFlag::eLIMIT_ENABLED, true);
    joint1->setLimit({-math::degToRad(120), math::degToRad(120)});

    joint2->setRevoluteJointFlag(physx::PxRevoluteJointFlag::eDRIVE_ENABLED, true);
    joint2->setRevoluteJointFlag(physx::PxRevoluteJointFlag::eLIMIT_ENABLED, true);
    joint2->setLimit({-math::degToRad(45), math::degToRad(45)});

    joint3->setRevoluteJointFlag(physx::PxRevoluteJointFlag::eDRIVE_ENABLED, true);
    joint3->setRevoluteJointFlag(physx::PxRevoluteJointFlag::eLIMIT_ENABLED, true);
    joint3->setLimit({-math::degToRad(90), math::degToRad(90)});

    KeyController keyListener({joint1, joint2, joint3});
    canvas.addKeyListener(keyListener);

    OrbitControls controls(camera, canvas);
    controls.enableKeys = false;

    bool run = false;
    KeyAdapter adapter(KeyAdapter::Mode::KEY_PRESSED, [&](KeyEvent) {
        run = true;
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
