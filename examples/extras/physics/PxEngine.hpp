
#ifndef THREEPP_PXENGINE_HPP
#define THREEPP_PXENGINE_HPP

#include "PxPhysicsAPI.h"

#include "threepp/threepp.hpp"

#include <unordered_map>

class PxEngine {

public:
    PxEngine(float timeStep = 1.f / 60)
        : timeStep(timeStep),
          sceneDesc(physics->getTolerancesScale()),
          onMeshRemovedListener(this) {

        sceneDesc.gravity = physx::PxVec3(0.0f, -9.81f, 0.0f);
        sceneDesc.flags |= physx::PxSceneFlag::eENABLE_PCM;

        if (!sceneDesc.cpuDispatcher) {
            cpuDispatcher = physx::PxDefaultCpuDispatcherCreate(1);
            sceneDesc.cpuDispatcher = cpuDispatcher;
        }
        if (!sceneDesc.filterShader)
            sceneDesc.filterShader = physx::PxDefaultSimulationFilterShader;
        scene = physics->createScene(sceneDesc);

        defaultMaterial = physics->createMaterial(0.5f, 0.5f, 0.6f);
    }

    void step(float dt) {

        internalTime += dt;

        while (internalTime >= timeStep) {

            scene->simulate(timeStep);
            scene->fetchResults(true);

            internalTime -= dt;
        }

        for (auto [obj, rb] : bodies) {

            auto t = rb->getGlobalPose();
            auto pos = t.p;
            auto quat = t.q;

            obj->position.set(pos.x, pos.y, pos.z);
            obj->quaternion.set(quat.x, quat.y, quat.z, quat.w);
        }

    }

    void registerMeshDynamic(threepp::Mesh& mesh) {

        threepp::Vector3 worldPos;
        mesh.getWorldPosition(worldPos);

        physx::PxTransform transform(toPxVector3(worldPos));
        auto geometry = toPxGeometry(mesh.geometry());
        if (!geometry) return;
        geometries[&mesh] = std::move(geometry);
        auto* actor = PxCreateDynamic(*physics, transform, *geometries[&mesh], *defaultMaterial, 10.0f);

        bodies[&mesh] = actor;
        scene->addActor(*actor);

        mesh.addEventListener("remove", &onMeshRemovedListener);
    }

    void registerMeshStatic(threepp::Mesh& mesh) {

        threepp::Vector3 worldPos;
        mesh.getWorldPosition(worldPos);

        physx::PxTransform transform(toPxVector3(worldPos));
        auto geometry = toPxGeometry(mesh.geometry());
        if (!geometry) return;
        geometries[&mesh] = std::move(geometry);
        auto* actor = PxCreateStatic(*physics, transform, *geometries[&mesh], *defaultMaterial);

        scene->addActor(*actor);
    }

    physx::PxActor* get(threepp::Mesh& mesh) {
        if (!bodies.count(&mesh)) return nullptr;
        return bodies.at(&mesh);
    }

    ~PxEngine() {

        for (auto [_, rb] : bodies) {
            rb->release();
        }

        defaultMaterial->release();
        if (cpuDispatcher) cpuDispatcher->release();
        scene->release();
        physics->release();
        foundation->release();
    }

private:
    float timeStep;
    float internalTime{};


    // Initialize the Physics SDK
    physx::PxDefaultAllocator allocator;
    physx::PxDefaultErrorCallback errorCallback;
    physx::PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errorCallback);
    physx::PxPhysics* physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, physx::PxTolerancesScale());

    physx::PxDefaultCpuDispatcher* cpuDispatcher;
    physx::PxSceneDesc sceneDesc;
    physx::PxScene* scene;
    physx::PxMaterial* defaultMaterial;

    std::unordered_map<threepp::Object3D*, std::unique_ptr<physx::PxGeometry>> geometries;
    std::unordered_map<threepp::Object3D*, physx::PxRigidBody*> bodies;


    static physx::PxVec3 toPxVector3(const threepp::Vector3& v) {
        return {v.x, v.y, v.z};
    }

    static physx::PxQuat toPxQuat(const threepp::Quaternion& q) {
        return {q.x, q.y, q.z, q.w};
    }

    static std::unique_ptr<physx::PxGeometry> toPxGeometry(const threepp::BufferGeometry* geometry) {

        if (!geometry) return nullptr;

        const auto type = geometry->type();
        if (type == "BoxGeometry") {
            const auto box = dynamic_cast<const threepp::BoxGeometry*>(geometry);
            return std::make_unique<physx::PxBoxGeometry>(physx::PxVec3{box->width / 2, box->height / 2, box->depth / 2});
        } else if (type == "SphereGeometry") {
            const auto sphere = dynamic_cast<const threepp::SphereGeometry*>(geometry);
            return std::make_unique<physx::PxSphereGeometry>(sphere->radius);
        } else {
            return nullptr;
        }
    }

    struct MeshRemovedListener: threepp::EventListener {

        explicit MeshRemovedListener(PxEngine* scope): scope(scope) {}

        void onEvent(threepp::Event& event) override {
            if (event.type == "remove") {
                auto m = static_cast<threepp::Mesh*>(event.target);
                if (scope->bodies.count(m)) {
                    auto rb = scope->bodies.at(m);
                    scope->scene->removeActor(*rb);
                    scope->bodies.erase(m);
                    scope->geometries.erase(m);
                }
                m->removeEventListener("remove", this);
            }
        }

    private:
        PxEngine* scope;
    };

    MeshRemovedListener onMeshRemovedListener;
};

#endif//THREEPP_PXENGINE_HPP
