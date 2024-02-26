
#ifndef THREEPP_PXENGINE_HPP
#define THREEPP_PXENGINE_HPP

#include "PxPhysicsAPI.h"

#include "threepp/threepp.hpp"

#include <unordered_map>

class PxEngine: public threepp::Object3D {

public:
    explicit PxEngine(float timeStep = 1.f / 60)
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
        scene->setVisualizationParameter(physx::PxVisualizationParameter::eSCALE, 1.0f);
        scene->setVisualizationParameter(physx::PxVisualizationParameter::eACTOR_AXES, 1.0f);
        scene->setVisualizationParameter(physx::PxVisualizationParameter::eJOINT_LOCAL_FRAMES, 2.0f);

        defaultMaterial = physics->createMaterial(0.5f, 0.5f, 0.6f);

        debugLines->material()->vertexColors = true;
        add(debugLines);
    }

    void step(float dt) {

        internalTime += dt;

        while (internalTime >= timeStep) {

            scene->simulate(timeStep);
            scene->fetchResults(true);

            internalTime -= dt;
        }

        threepp::Matrix4 tmpMat;
        threepp::Quaternion tmpQuat;
        for (auto [obj, rb] : bodies) {

            auto t = rb->getGlobalPose();
            auto pos = t.p;
            auto quat = t.q;

            obj->position.set(pos.x, pos.y, pos.z);
            obj->applyMatrix4(tmpMat.copy(*obj->parent->matrix).invert());
            obj->quaternion.set(quat.x, quat.y, quat.z, quat.w);
            obj->applyQuaternion(tmpQuat.copy(obj->parent->quaternion).invert());
        }

        std::vector<float> lineVertices;
        std::vector<float> lineColors;
        const auto& rb = scene->getRenderBuffer();
        for (auto i = 0; i < rb.getNbLines(); i++) {
            const auto& line = rb.getLines()[i];

            lineVertices.insert(lineVertices.end(), {line.pos0.x, line.pos0.y, line.pos0.z, line.pos1.x, line.pos1.y, line.pos1.z});
            threepp::Color c1 = line.color0;
            threepp::Color c2 = line.color1;
            lineColors.insert(lineColors.end(), {c1.r, c1.g, c1.b, c2.r, c2.g, c2.b});
        }

        if (!lineVertices.empty()) {
            auto geom = threepp::BufferGeometry::create();
            geom->setAttribute("position", threepp::FloatBufferAttribute::create(lineVertices, 3));
            geom->setAttribute("color", threepp::FloatBufferAttribute::create(lineColors, 3));
            debugLines->setGeometry(geom);
        }
    }

    void registerMeshDynamic(threepp::Object3D& obj) {

        threepp::Vector3 worldPos;
        obj.getWorldPosition(worldPos);

        physx::PxTransform transform(toPxVector3(worldPos));
        auto geometry = toPxGeometry(obj.geometry());
        if (!geometry) return;
        geometries[&obj] = std::move(geometry);
        auto* actor = PxCreateDynamic(*physics, transform, *geometries[&obj], *defaultMaterial, 10.0f);

        bodies[&obj] = actor;
        scene->addActor(*actor);

        obj.addEventListener("remove", &onMeshRemovedListener);
    }

    void registerMeshStatic(threepp::Object3D& obj) {

        threepp::Vector3 worldPos;
        obj.getWorldPosition(worldPos);

        physx::PxTransform transform(toPxVector3(worldPos));
        auto geometry = toPxGeometry(obj.geometry());
        if (!geometry) return;
        geometries[&obj] = std::move(geometry);
        auto* actor = PxCreateStatic(*physics, transform, *geometries[&obj], *defaultMaterial);

        bodies[&obj] = actor;
        scene->addActor(*actor);

        obj.addEventListener("remove", &onMeshRemovedListener);
    }

    physx::PxRevoluteJoint* createRevoluteJoint(threepp::Object3D& o1, threepp::Vector3 anchor, threepp::Vector3 axis) {

        auto rb1 = bodies.at(&o1);

        threepp::Matrix4 f1;
        f1.makeRotationFromQuaternion(threepp::Quaternion().setFromUnitVectors({1,0,0}, axis));
        f1.setPosition(anchor);

        threepp::Matrix4 f2;
        f2.makeRotationFromQuaternion(threepp::Quaternion().setFromUnitVectors({1,0,0}, axis));

        physx::PxTransform frame1 = toPxTransform(f1);
        physx::PxTransform frame2 = toPxTransform(f2);

        physx::PxRevoluteJoint* joint = physx::PxRevoluteJointCreate(*physics, rb1, frame1, nullptr, frame2);
        joint->setConstraintFlag(physx::PxConstraintFlag::eVISUALIZATION, true);

        return joint;
    }

    physx::PxRevoluteJoint* createRevoluteJoint(threepp::Object3D& o1, threepp::Object3D& o2, threepp::Vector3 anchor, threepp::Vector3 axis) {

        o1.updateMatrixWorld();
        o2.updateMatrixWorld();

        auto rb1 = bodies.at(&o1);
        auto rb2 = bodies.at(&o2);

        threepp::Matrix4 f1;
        f1.makeRotationFromQuaternion(threepp::Quaternion().setFromUnitVectors({1, 0, 0}, axis));
        f1.setPosition(anchor);

        threepp::Matrix4 f2 = *o2.matrixWorld;
        f2.invert().multiply(*o1.matrixWorld).multiply(f1);

        physx::PxTransform frame1 = toPxTransform(f1);
        physx::PxTransform frame2 = toPxTransform(f2);

        physx::PxRevoluteJoint* joint = physx::PxRevoluteJointCreate(*physics, rb1, frame1, rb2, frame2);
        joint->setConstraintFlag(physx::PxConstraintFlag::eVISUALIZATION, true);

        return joint;
    }

    physx::PxActor* get(threepp::Mesh& mesh) {
        if (!bodies.count(&mesh)) return nullptr;
        return bodies.at(&mesh);
    }

    ~PxEngine() override {

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
    std::unordered_map<threepp::Object3D*, physx::PxRigidActor*> bodies;

    std::shared_ptr<threepp::LineSegments> debugLines = threepp::LineSegments::create();


    static physx::PxVec3 toPxVector3(const threepp::Vector3& v) {
        return {v.x, v.y, v.z};
    }

    static physx::PxQuat toPxQuat(const threepp::Quaternion& q) {
        return {q.x, q.y, q.z, q.w};
    }

    static physx::PxTransform toPxTransform(threepp::Matrix4& m) {
        threepp::Vector3 pos;
        threepp::Quaternion quat;
        threepp::Vector3 scale;

        m.decompose(pos, quat, scale);

        return physx::PxTransform(toPxVector3(pos), toPxQuat(quat));
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
