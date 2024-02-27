
#ifndef THREEPP_PXENGINE_HPP
#define THREEPP_PXENGINE_HPP

#include "PxPhysicsAPI.h"

#include "threepp/threepp.hpp"

#include <unordered_map>

class PxEngine: public threepp::Object3D {

public:
    explicit PxEngine(float timeStep = 1.f / 100)
        : timeStep(timeStep),
          sceneDesc(physics->getTolerancesScale()),
          onMeshRemovedListener(this) {

        sceneDesc.gravity = physx::PxVec3(0.0f, -9.81f, 0.0f);
        sceneDesc.flags |= physx::PxSceneFlag::eENABLE_PCM;

        if (!sceneDesc.cpuDispatcher) {
            cpuDispatcher = physx::PxDefaultCpuDispatcherCreate(1);
            sceneDesc.cpuDispatcher = cpuDispatcher;
        }
        if (!sceneDesc.filterShader) {
            sceneDesc.filterShader = physx::PxDefaultSimulationFilterShader;
        }

        sceneDesc.solverType = physx::PxSolverType::eTGS;

        scene = physics->createScene(sceneDesc);
        scene->setVisualizationParameter(physx::PxVisualizationParameter::eSCALE, 1.0f);
        scene->setVisualizationParameter(physx::PxVisualizationParameter::eACTOR_AXES, 1.0f);
        scene->setVisualizationParameter(physx::PxVisualizationParameter::eJOINT_LOCAL_FRAMES, 1.0f);
        scene->setVisualizationParameter(physx::PxVisualizationParameter::eCONTACT_POINT, 2.0f);
        scene->setVisualizationParameter(physx::PxVisualizationParameter::eJOINT_LIMITS, 1.0f);
        scene->setVisualizationParameter(physx::PxVisualizationParameter::eCONTACT_FORCE, 1.0f);
        scene->setVisualizationParameter(physx::PxVisualizationParameter::eCOLLISION_SHAPES, 1.0f);

        defaultMaterial = physics->createMaterial(0.5f, 0.5f, 0.6f);

        debugLines->material()->vertexColors = true;
        debugPoints->material()->vertexColors = true;
        debugTriangles->material()->vertexColors = true;
        add(debugLines);
        add(debugPoints);
        add(debugTriangles);
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

            obj->updateMatrixWorld();

            auto t = rb->getGlobalPose();
            auto pos = t.p;
            auto quat = t.q;

            tmpQuat.set(quat.x, quat.y, quat.z, quat.w);
            obj->matrix->makeRotationFromQuaternion(tmpQuat);
            obj->matrix->setPosition(pos.x, pos.y, pos.z);
            obj->matrix->premultiply(tmpMat.copy(*obj->parent->matrixWorld).invert());
        }

        debugRender();
    }

    physx::PxRigidDynamic* registerMeshDynamic(threepp::Object3D& obj) {

        threepp::Vector3 worldPos;
        obj.getWorldPosition(worldPos);

        physx::PxTransform transform(toPxVector3(worldPos));
        auto geometry = toPxGeometry(obj.geometry());
        if (!geometry) return nullptr;
        geometries[&obj] = std::move(geometry);
        auto* actor = PxCreateDynamic(*physics, transform, *geometries[&obj], *defaultMaterial, 10.0f);

        bodies[&obj] = actor;
        scene->addActor(*actor);

        actor->setSolverIterationCounts(30, 2);

        obj.matrixAutoUpdate = false;

        obj.addEventListener("remove", &onMeshRemovedListener);

        return actor;
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
        f1.makeRotationFromQuaternion(threepp::Quaternion().setFromUnitVectors({1, 0, 0}, axis));
        f1.setPosition(anchor);

        threepp::Matrix4 f2 = *o1.matrixWorld;
        f2.invert().multiply(f1);

        physx::PxTransform frame1 = toPxTransform(f1);
        physx::PxTransform frame2 = toPxTransform(f2);

        physx::PxRevoluteJoint* joint = physx::PxRevoluteJointCreate(*physics, nullptr, frame1, rb1, frame2);
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

    [[nodiscard]] physx::PxScene* getScene() const {
        return scene;
    }

    [[nodiscard]] physx::PxPhysics* getPhysics() const {
        return physics;
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

    std::shared_ptr<threepp::Mesh> debugTriangles = threepp::Mesh::create();
    std::shared_ptr<threepp::LineSegments> debugLines = threepp::LineSegments::create();
    std::shared_ptr<threepp::Points> debugPoints = threepp::Points::create();

    void debugRender() {
        std::vector<float> lineVertices;
        std::vector<float> lineColors;
        const auto& buffer = scene->getRenderBuffer();
        for (auto i = 0; i < buffer.getNbLines(); i++) {
            const auto& line = buffer.getLines()[i];

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

        std::vector<float> triangleVertices;
        std::vector<float> triangleColors;
        for (auto i = 0; i < buffer.getNbTriangles(); i++) {
            const auto& point = buffer.getTriangles()[i];
            auto pos1 = point.pos0;
            auto pos2 = point.pos1;
            auto pos3 = point.pos2;
            threepp::Color color1 = point.color0;
            threepp::Color color2 = point.color1;
            threepp::Color color3 = point.color2;
            triangleVertices.insert(triangleVertices.end(), {pos1.x, pos1.y, pos1.z, pos2.x, pos2.y, pos2.z, pos3.x, pos3.y, pos3.z});
            triangleColors.insert(triangleColors.end(), {color1.r, color1.g, color1.b, color2.r, color2.g, color2.b, color3.r, color3.g, color3.b});
        }
        if (!triangleVertices.empty()) {
            auto geom = threepp::BufferGeometry::create();
            geom->setAttribute("position", threepp::FloatBufferAttribute::create(triangleVertices, 3));
            geom->setAttribute("color", threepp::FloatBufferAttribute::create(triangleColors, 3));
            debugTriangles->setGeometry(geom);
        }

        std::vector<float> pointVertices;
        std::vector<float> pointColors;
        for (auto i = 0; i < buffer.getNbPoints(); i++) {
            const auto& point = buffer.getPoints()[i];
            auto pos = point.pos;
            threepp::Color color = point.color;
            pointVertices.insert(pointVertices.end(), {pos.x, pos.y, pos.z});
            pointColors.insert(pointColors.end(), {color.r, color.g, color.b});
        }
        if (!pointVertices.empty()) {
            auto geom = threepp::BufferGeometry::create();
            geom->setAttribute("position", threepp::FloatBufferAttribute::create(pointVertices, 3));
            geom->setAttribute("color", threepp::FloatBufferAttribute::create(pointColors, 3));
            debugPoints->setGeometry(geom);
        }
    }

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

        return {toPxVector3(pos), toPxQuat(quat)};
    }

    std::unique_ptr<physx::PxGeometry> toPxGeometry(const threepp::BufferGeometry* geometry) {

        if (!geometry) return nullptr;

        const auto type = geometry->type();
        if (type == "BoxGeometry") {
            const auto box = dynamic_cast<const threepp::BoxGeometry*>(geometry);
            return std::make_unique<physx::PxBoxGeometry>(physx::PxVec3{box->width / 2, box->height / 2, box->depth / 2});
        } else if (type == "SphereGeometry") {
            const auto sphere = dynamic_cast<const threepp::SphereGeometry*>(geometry);
            return std::make_unique<physx::PxSphereGeometry>(sphere->radius);
        } else if (type == "CapsuleGeometry") {
            const auto cap = dynamic_cast<const threepp::CapsuleGeometry*>(geometry);
            return std::make_unique<physx::PxCapsuleGeometry>(cap->radius, cap->length / 2);
        } else {

            auto pos = geometry->getAttribute<float>("position");
            if (pos) {
                physx::PxConvexMeshDesc convexDesc;
                convexDesc.points.count = 50;
                convexDesc.points.stride = sizeof(physx::PxVec3);
                convexDesc.points.data = pos->array().data();
                convexDesc.flags = physx::PxConvexFlag::eCOMPUTE_CONVEX;

                physx::PxTolerancesScale scale;
                physx::PxCookingParams params(scale);

                physx::PxDefaultMemoryOutputStream buf;
                physx::PxConvexMeshCookingResult::Enum result;
                if (!PxCookConvexMesh(params, convexDesc, buf, &result)) {
                    return nullptr;
                }
                physx::PxDefaultMemoryInputData input(buf.getData(), buf.getSize());
                auto convexMesh = physics->createConvexMesh(input);

                return std::make_unique<physx::PxConvexMeshGeometry>(convexMesh);
            }

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
