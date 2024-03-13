
#ifndef THREEPP_PXENGINE_HPP
#define THREEPP_PXENGINE_HPP

#include "PxPhysicsAPI.h"

#include "RigidbodyInfo.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/CapsuleGeometry.hpp"
#include "threepp/geometries/CylinderGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Points.hpp"

#include <functional>
#include <unordered_map>

namespace {

    using JointCreate = std::function<physx::PxJoint*(physx::PxPhysics& physics, physx::PxRigidActor* actor0, const physx::PxTransform& localFrame0, physx::PxRigidActor* actor1, const physx::PxTransform& localFrame1)>;

    physx::PxVec3 toPxVector3(const threepp::Vector3& v) {
        return {v.x, v.y, v.z};
    }

    physx::PxQuat toPxQuat(const threepp::Quaternion& q) {
        return {q.x, q.y, q.z, q.w};
    }

    physx::PxTransform toPxTransform(const threepp::Matrix4& m) {
        threepp::Vector3 pos;
        threepp::Quaternion quat;
        threepp::Vector3 scale;

        m.decompose(pos, quat, scale);

        return {toPxVector3(pos), toPxQuat(quat)};
    }

}// namespace

class PxEngine: public threepp::Object3D {

public:
    bool debugVisualisation = true;

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

    PxEngine(const PxEngine&) = delete;
    PxEngine(PxEngine&&) = delete;
    PxEngine& operator=(const PxEngine&) = delete;
    PxEngine& operator=(PxEngine&&) = delete;

    void setup(threepp::Object3D& obj) {

        obj.traverse([this](threepp::Object3D& obj) {
            if (obj.userData.count("rigidbodyInfo")) {
                auto info = std::any_cast<threepp::RigidBodyInfo>(obj.userData.at("rigidbodyInfo"));
                registerObject(obj, info);
            }
        });

        obj.traverse([this](threepp::Object3D& obj) {
            if (obj.userData.count("rigidbodyInfo")) {
                auto info = std::any_cast<threepp::RigidBodyInfo>(obj.userData.at("rigidbodyInfo"));
                if (info._joint) {
                    physx::PxJoint* joint = nullptr;
                    switch (info._joint->type) {
                        case threepp::JointInfo::Type::HINGE:
                            joint = createJoint(physx::PxRevoluteJointCreate, &obj, info._joint->connectedBody, info._joint->anchor, info._joint->axis);
                            if (info._joint->limits) {
                                joint->is<physx::PxRevoluteJoint>()->setRevoluteJointFlag(physx::PxRevoluteJointFlag::eLIMIT_ENABLED, true);
                                joint->is<physx::PxRevoluteJoint>()->setLimit({info._joint->limits->x, info._joint->limits->y});
                            }
                            break;
                        case threepp::JointInfo::Type::PRISMATIC:
                            joint = createJoint(physx::PxDistanceJointCreate, &obj, info._joint->connectedBody, info._joint->anchor, info._joint->axis);
                            break;
                        case threepp::JointInfo::Type::BALL:
                            joint = createJoint(physx::PxSphericalJointCreate, &obj, info._joint->connectedBody, info._joint->anchor, info._joint->axis);
                            break;
                        case threepp::JointInfo::Type::LOCK:
                            joint = createJoint(physx::PxFixedJointCreate, &obj, info._joint->connectedBody, info._joint->anchor, info._joint->axis);
                            break;
                    }
                    if (joint) {
                        joints[&obj].push_back(joint);
                    }
                }
            }
        });
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

        if (debugVisualisation) {
            visible = true;
            debugRender();
        } else {
            visible = false;
        }
    }

    void registerObject(threepp::Object3D& obj, const threepp::RigidBodyInfo& info) {

        if (!info._useVisualGeometryAsCollider && info._colliders.empty()) {
            return;
        }
        if (info._useVisualGeometryAsCollider && !obj.geometry()) {
            return;
        }

        obj.updateMatrixWorld();

        std::vector<physx::PxShape*> shapes;

        if (info._useVisualGeometryAsCollider) {
            auto shape = toPxShape(obj.geometry());
            if (!shape) {
                shape = physics->createShape(physx::PxSphereGeometry(0.1), *defaultMaterial, true);//dummy
            }
            shapes.emplace_back(shape);
        }

        for (const auto& [collider, offset] : info._colliders) {
            auto shape = toPxShape(collider);
            shape->setLocalPose(toPxTransform(offset));
            shapes.emplace_back(shape);
        }

        if (info._type == threepp::RigidBodyInfo::Type::STATIC) {
            auto staticActor = PxCreateStatic(*physics, toPxTransform(*obj.matrixWorld), *shapes.front());

            for (unsigned i = 1; i < shapes.size(); i++) {
                auto shape = shapes[i];
                staticActor->attachShape(*shape);
            }

            bodies[&obj] = staticActor;
        } else {
            auto rigidActor = PxCreateDynamic(*physics, toPxTransform(*obj.matrixWorld), *shapes.front(), 1.f);

            rigidActor->setSolverIterationCounts(30, 2);

            for (unsigned i = 1; i < shapes.size(); i++) {
                auto shape = shapes[i];
                rigidActor->attachShape(*shape);
            }

            if (info._mass) {
                physx::PxRigidBodyExt::setMassAndUpdateInertia(*rigidActor, *info._mass);
            } else {
                physx::PxRigidBodyExt::updateMassAndInertia(*rigidActor, 1.f);
            }

            bodies[&obj] = rigidActor;
        }

        for (auto shape : shapes) {
            shape->release();
        }

        scene->addActor(*bodies[&obj]);

        obj.matrixAutoUpdate = false;
        obj.addEventListener("remove", &onMeshRemovedListener);
    }

    physx::PxJoint* createJoint(const JointCreate& create, threepp::Object3D* o1, threepp::Object3D* o2, threepp::Vector3 anchor, threepp::Vector3 axis) {

        o1->updateMatrixWorld();
        if (o2) o2->updateMatrixWorld();

        auto rb1 = bodies.at(o1);
        auto rb2 = o2 ? bodies.at(o2) : nullptr;

        threepp::Matrix4 f1;
        f1.makeRotationFromQuaternion(threepp::Quaternion().setFromUnitVectors({1, 0, 0}, axis));
        f1.setPosition(anchor);

        threepp::Matrix4 f2 = o2 ? *o2->matrixWorld : threepp::Matrix4();
        f2.invert().multiply(*o1->matrixWorld).multiply(f1);

        physx::PxTransform frame1 = toPxTransform(f1);
        physx::PxTransform frame2 = toPxTransform(f2);

        auto* joint = create(*physics, rb2, frame2, rb1, frame1);
        joint->setConstraintFlag(physx::PxConstraintFlag::eVISUALIZATION, true);

        return joint;
    }

    physx::PxRigidActor* getBody(threepp::Object3D& mesh) {
        if (!bodies.count(&mesh)) return nullptr;
        return bodies.at(&mesh);
    }

    template<class JointType>
    JointType* getJoint(threepp::Object3D& obj, int index = 0) {
        if (!joints.count(&obj)) return nullptr;
        auto* joint = joints.at(&obj).at(index);
        return joint->is<JointType>();
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

    std::unordered_map<threepp::Object3D*, physx::PxRigidActor*> bodies;
    std::unordered_map<threepp::Object3D*, std::vector<physx::PxJoint*>> joints;

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

    physx::PxShape* toPxShape(const threepp::Collider& collider) {

        switch (collider.index()) {
            case 0:// Sphere
            {
                threepp::SphereCollider sphere = std::get<threepp::SphereCollider>(collider);
                return physics->createShape(physx::PxSphereGeometry(sphere.radius), *defaultMaterial, true);
            }
            case 1:// Box
            {
                threepp::BoxCollider box = std::get<threepp::BoxCollider>(collider);
                return physics->createShape(physx::PxBoxGeometry(box.halfWidth, box.halfHeight, box.halfDepth), *defaultMaterial, true);
            }
            case 2:// Capsule
            {
                threepp::CapsuleCollider box = std::get<threepp::CapsuleCollider>(collider);
                return physics->createShape(physx::PxCapsuleGeometry(box.radius, box.halfHeight), *defaultMaterial, true);
            }
            default:
                return nullptr;
        }
    }

    physx::PxShape* toPxShape(const threepp::BufferGeometry* geometry) {

        if (!geometry) return nullptr;

        const auto type = geometry->type();
        if (type == "BoxGeometry") {
            const auto box = dynamic_cast<const threepp::BoxGeometry*>(geometry);
            return physics->createShape(physx::PxBoxGeometry(physx::PxVec3{box->width / 2, box->height / 2, box->depth / 2}), *defaultMaterial, true);
        } else if (type == "SphereGeometry") {
            const auto sphere = dynamic_cast<const threepp::SphereGeometry*>(geometry);
            return physics->createShape(physx::PxSphereGeometry(sphere->radius), *defaultMaterial, true);
        } else if (type == "CapsuleGeometry") {
            const auto cap = dynamic_cast<const threepp::CapsuleGeometry*>(geometry);
            return physics->createShape(physx::PxCapsuleGeometry(cap->radius, cap->length / 2), *defaultMaterial, true);
        } else {

            auto pos = geometry->getAttribute<float>("position");
            if (pos) {
                physx::PxConvexMeshDesc convexDesc;
                convexDesc.points.count = pos->count() / 2;
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

                return physics->createShape(physx::PxConvexMeshGeometry(convexMesh), *defaultMaterial, true);
            }

            return nullptr;
        }
    }

    struct MeshRemovedListener: threepp::EventListener {

        explicit MeshRemovedListener(PxEngine* scope): scope(scope) {}

        void onEvent(threepp::Event& event) override {
            if (event.type == "remove") {
                auto m = static_cast<threepp::Object3D*>(event.target);
                if (scope->bodies.count(m)) {
                    auto rb = scope->bodies.at(m);
                    scope->scene->removeActor(*rb);
                    scope->bodies.erase(m);
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
