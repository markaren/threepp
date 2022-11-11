
#include "threepp/threepp.hpp"

#include <bullet/btBulletDynamicsCommon.h>

using namespace threepp;

btVector3 convert (const Vector3& v) {
    return { v.x, v.y, v.z };
}

int main() {

    ///collision configuration contains default setup for memory, collision setup
    auto collisionConfiguration = btDefaultCollisionConfiguration();

    ///use the default collision dispatcher. For parallel processing you can use a diffent dispatcher (see Extras/BulletMultiThreaded)
    auto dispatcher = btCollisionDispatcher(&collisionConfiguration);

    auto broadphase = btDbvtBroadphase();

    ///the default constraint solver. For parallel processing you can use a different solver (see Extras/BulletMultiThreaded)
    auto solver = btSequentialImpulseConstraintSolver();

    auto world = std::make_unique<btDiscreteDynamicsWorld>(&dispatcher, &broadphase, &solver, &collisionConfiguration);
    world->setGravity(btVector3(0, -10, 0));

    Canvas canvas;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.set(-5,5,5);

    OrbitControls controls{camera, canvas};

    GLRenderer renderer(canvas);
    renderer.setClearColor(Color(Color::aliceblue));
    renderer.setSize(canvas.getSize());

    const auto boxGeometry = BoxGeometry::create();
    const auto boxMaterial = MeshBasicMaterial::create();
    boxMaterial->color.setRGB(1,0,0);
    auto box = Mesh::create(boxGeometry, boxMaterial);
    box->position.setY(5);
    scene->add(box);

    const auto planeGeometry = PlaneGeometry::create(10, 10);
    planeGeometry->rotateX(math::degToRad(-90));
    const auto planeMaterial = MeshBasicMaterial::create();
    planeMaterial->color.setRGB(0,0,1);
    auto plane = Mesh::create(planeGeometry, planeMaterial);
    scene->add(plane);


    canvas.onWindowResize([&](WindowSize size){
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    auto b1 = new btBoxShape(btVector3(boxGeometry->width, boxGeometry->height, boxGeometry->width)/2);
    auto t1 = new btDefaultMotionState();
    auto t = btTransform();
    t.setOrigin(convert(box->position));
    t1->setWorldTransform(t);
    auto boxBody = btRigidBody(1, t1, b1);
    world->addRigidBody(&boxBody);

    auto b2 = new btBoxShape(btVector3(planeGeometry->width, 0.1, boxGeometry->height)/2);
    auto t2 = new btDefaultMotionState();
    auto planeBody = btRigidBody(0, t2, b2);
    world->addRigidBody(&planeBody);

    box->rotation.setOrder(Euler::YZX);
    canvas.animate([&](float dt) {

        world->stepSimulation(dt);

        auto p = boxBody.getWorldTransform().getOrigin();
        box->position.set(p.x(), p.y(), p.z());

        renderer.render(scene, camera);
    });
}