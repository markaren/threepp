
#include "PxEngine.hpp"


using namespace threepp;

namespace {

    struct MyKeyListener: KeyListener {

        explicit MyKeyListener(physx::PxRevoluteJoint* j1, physx::PxRevoluteJoint* j2): j1(j1), j2(j2) {}

        void onKeyPressed(KeyEvent evt) override {
            if (evt.key == Key::NUM_1) {
                j1->setDriveVelocity(-5);// Set motor speed
            } else if (evt.key == Key::NUM_2) {
                j1->setDriveVelocity(5);// Set motor speed
            }
            if (evt.key == Key::NUM_3) {
                j2->setDriveVelocity(-5);// Set motor speed
            } else if (evt.key == Key::NUM_4) {
                j2->setDriveVelocity(5);// Set motor speed
            }
        }
        void onKeyReleased(KeyEvent evt) override {
            switch (evt.key) {
                case Key::NUM_1:
                case Key::NUM_2: {
                    j1->setDriveVelocity(0);// Set motor speed
                }

                case Key::NUM_3:
                case Key::NUM_4: {
                    j2->setDriveVelocity(0);// Set motor speed
                }
            }
        }

    private:
        float speed = 5;
        physx::PxRevoluteJoint* j1;
        physx::PxRevoluteJoint* j2;
    };

}// namespace

int main() {

    Canvas canvas("PhysX");
    GLRenderer renderer(canvas.size());

    Scene scene;
    scene.background = Color::navy;

    PerspectiveCamera camera(60, canvas.aspect());
    camera.position.z = 10;

    auto box1 = Mesh::create(BoxGeometry::create(0.5, 2, 0.5), MeshStandardMaterial::create());
    box1->position.y = 1;
    scene.add(box1);

    auto box2 = Mesh::create(BoxGeometry::create(1, 0.5, 0.5), MeshStandardMaterial::create());
    box2->position.y = 2 + 0.25;
    box2->position.x = 0.5 + 0.25;
    scene.add(box2);

    auto ground = Mesh::create(BoxGeometry::create(10, 0.1, 10), MeshStandardMaterial::create({{"color", Color::blueviolet}}));
    scene.add(ground);

    auto light = HemisphereLight::create(Color::brown, Color::blanchedalmond);
    light->position.y = 5;
    scene.add(light);

    PxEngine engine;
    engine.registerMeshDynamic(*box1);
    engine.registerMeshDynamic(*box2);
    engine.registerMeshStatic(*ground);
    auto joint1 = engine.createRevoluteJoint(*ground, *box1, {0, 0, 0}, {0, 1, 0});
    auto joint2 = engine.createRevoluteJoint(*box1, *box2, {0.25, 1, 0}, {1, 0, 0});

    joint1->setRevoluteJointFlag(physx::PxRevoluteJointFlag::eDRIVE_ENABLED, true);
    joint1->setRevoluteJointFlag(physx::PxRevoluteJointFlag::eLIMIT_ENABLED, true);
    joint1->setLimit(physx::PxJointAngularLimitPair(-math::degToRad(90), math::degToRad(90)));

    joint2->setRevoluteJointFlag(physx::PxRevoluteJointFlag::eDRIVE_ENABLED, true);
    joint2->setRevoluteJointFlag(physx::PxRevoluteJointFlag::eLIMIT_ENABLED, true);
    joint2->setLimit(physx::PxJointAngularLimitPair(-math::degToRad(45), math::degToRad(45)));

    MyKeyListener keyListener(joint1, joint2);
    canvas.addKeyListener(keyListener);

    OrbitControls controls(camera, canvas);
    controls.enableKeys = false;

    Clock clock;
    canvas.animate([&] {
        auto dt = clock.getDelta();

        renderer.render(scene, camera);

        engine.step(dt);
    });
}
