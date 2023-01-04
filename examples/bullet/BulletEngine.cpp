
#include "BulletEngine.hpp"

#include <threepp/geometries/geometries.hpp>

#include <unordered_map>

using namespace threepp;

namespace {

    std::unique_ptr<btCollisionShape> convert(const std::shared_ptr<BufferGeometry> &geometry) {

        if (std::dynamic_pointer_cast<const BoxGeometry>(geometry)) {

            auto g = std::dynamic_pointer_cast<const BoxGeometry>(geometry);
            return std::make_unique<btBoxShape>(btVector3(g->width, g->height, g->depth) / 2);

        } else if (std::dynamic_pointer_cast<const PlaneGeometry>(geometry)) {

            auto g = std::dynamic_pointer_cast<const PlaneGeometry>(geometry);
            return std::make_unique<btBoxShape>(btVector3(g->width / 2, 0.01, g->height / 2));

        } else if (std::dynamic_pointer_cast<const SphereGeometry>(geometry)) {

            auto g = std::dynamic_pointer_cast<const SphereGeometry>(geometry);
            return std::make_unique<btSphereShape>(g->radius);

        } else if (std::dynamic_pointer_cast<const CylinderGeometry>(geometry)) {

            auto g = std::dynamic_pointer_cast<const CylinderGeometry>(geometry);
            return std::make_unique<btCylinderShape>(btVector3(g->radiusTop, g->height / 2, g->radiusTop));
        }

        return nullptr;
    }

    std::unique_ptr<Rbinfo> createFromMesh(Mesh *m, float mass) {
        m->updateMatrixWorld();
        auto collisionShape = convert(m->geometry());
        if (!collisionShape) return nullptr;
        return std::make_unique<Rbinfo>(std::move(collisionShape), mass, *m->matrixWorld);
    }

    std::unique_ptr<Rbinfo> createFromMesh(const std::vector<Mesh *> &meshes, float mass) {
        if (meshes.empty()) return nullptr;
        if (meshes.size() == 1) return createFromMesh(meshes.back(), mass);

        auto collisionShape = std::make_unique<btCompoundShape>();
        for (auto &m : meshes) {
            m->updateMatrixWorld();
            auto shape = convert(m->geometry());
            collisionShape->addChildShape(convert(*m->matrixWorld), shape.get());
        }
        return std::make_unique<Rbinfo>(std::move(collisionShape), mass, Matrix4());
    }

}// namespace


btRigidBody *BulletEngine::registerMesh(const std::shared_ptr<Mesh> &m, float mass) {

    auto rb = createFromMesh(m.get(), mass);
    if (!rb) return nullptr;

    world.addRigidBody(rb->body.get());
    bodies[m] = std::move(rb);
    return bodies.at(m)->body.get();
}

btRigidBody *BulletEngine::registerGroup(const std::shared_ptr<threepp::Group> &m, float mass) {

    std::vector<Mesh *> meshes;
    m->traverseType<Mesh>([&](Mesh &o) {
        meshes.emplace_back(&o);
    });

    auto rb = createFromMesh(meshes, mass);
    if (!rb) return nullptr;

    world.addRigidBody(rb->body.get());
    bodies[m] = std::move(rb);
    return bodies.at(m)->body.get();
}

void BulletEngine::step(float dt) {

    world.stepSimulation(dt, 1, 1.0f/100);

    for (auto &[m, info] : bodies) {
        auto t = info->body->getWorldTransform();
        auto p = t.getOrigin();
        auto r = t.getRotation();

        m->quaternion.set(r.x(), r.y(), r.z(), r.w());
        m->position.set(p.x(), p.y(), p.z());
    }
}

void BulletEngine::addConstraint(btTypedConstraint* c) {
    world.addConstraint(c, true);
}
