
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
#include "threepp/objects/InstancedMesh.hpp"
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
        return physx::PxTransform(physx::PxMat44((float*) m.elements.data()));
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
        scene->setVisualizationParameter(physx::PxVisualizationParameter::eACTOR_AXES, 1.0f);
        scene->setVisualizationParameter(physx::PxVisualizationParameter::eJOINT_LOCAL_FRAMES, 0.5f);
        scene->setVisualizationParameter(physx::PxVisualizationParameter::eCONTACT_POINT, 2.0f);
        scene->setVisualizationParameter(physx::PxVisualizationParameter::eJOINT_LIMITS, 1.0f);
        scene->setVisualizationParameter(physx::PxVisualizationParameter::eCONTACT_FORCE, 1.0f);
        scene->setVisualizationParameter(physx::PxVisualizationParameter::eCOLLISION_SHAPES, 1.0f);

        materials.emplace_back(physics->createMaterial(0.5f, 0.5f, 0.6f));

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
                const auto info = std::any_cast<threepp::RigidBodyInfo>(obj.userData.at("rigidbodyInfo"));
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

            internalTime -= timeStep;
        }

        static threepp::Matrix4 tmpMat;
        for (auto& [obj, rb] : bodies) {

            obj->updateMatrixWorld();

            if (auto instanced = obj->as<threepp::InstancedMesh>()) {

                tmpMat.copy(*obj->matrix).invert();
                auto& array = instanced->instanceMatrix()->array();
                for (int i = 0; i < instanced->count(); i++) {
                    auto pose = threepp::Matrix4().fromArray(physx::PxMat44(rb[i]->getGlobalPose()).front());
                    pose.premultiply(tmpMat);
                    for (auto j = 0; j < 16; j++) {
                        array[i * 16 + j] = pose[j];
                    }
                }

                instanced->instanceMatrix()->needsUpdate();

            } else {

                const auto pose = physx::PxMat44(rb.front()->getGlobalPose());
                obj->matrix->fromArray(pose.front());
            }
            obj->matrix->premultiply(tmpMat.copy(*obj->parent->matrixWorld).invert());
        }

        if (debugVisualisation) {
            visible = true;
            scene->setVisualizationParameter(physx::PxVisualizationParameter::eSCALE, 1.0f);
            debugRender();
        } else {
            scene->setVisualizationParameter(physx::PxVisualizationParameter::eSCALE, 0);
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

        const auto material = toPxMaterial(info._material);

        std::vector<physx::PxShape*> shapes;

        if (info._useVisualGeometryAsCollider) {
            auto shape = toPxShape(obj.geometry(), material);
            if (!shape) {
                shape = physics->createShape(physx::PxSphereGeometry(0.1), *material, true);//dummy
            }
            shapes.emplace_back(shape);
        }

        for (const auto& [collider, offset] : info._colliders) {
            auto shape = toPxShape(collider, material);
            shape->setLocalPose(toPxTransform(offset));
            shapes.emplace_back(shape);
        }


        if (info._type == threepp::RigidBodyInfo::Type::STATIC) {
            auto staticActor = PxCreateStatic(*physics, toPxTransform(*obj.matrixWorld), *shapes.front());

            for (unsigned i = 1; i < shapes.size(); i++) {
                staticActor->attachShape(*shapes[i]);
            }

            bodies[&obj].emplace_back(staticActor);
        } else {

            auto createSingle = [&](float* data) {
                auto rigidActor = PxCreateDynamic(*physics, physx::PxTransform(physx::PxMat44(data)), *shapes.front(), 1.f);
                rigidActor->setSolverIterationCounts(30, 2);

                for (unsigned i = 1; i < shapes.size(); i++) {
                    rigidActor->attachShape(*shapes[i]);
                }

                if (info._mass) {
                    physx::PxRigidBodyExt::setMassAndUpdateInertia(*rigidActor, *info._mass);
                } else {
                    physx::PxRigidBodyExt::updateMassAndInertia(*rigidActor, 1.f);
                }
                return rigidActor;
            };

            if (auto instanced = obj.as<threepp::InstancedMesh>()) {

                instanced->frustumCulled = false;

                const auto& array = instanced->instanceMatrix()->array();

                threepp::Matrix4 tmp;
                for (unsigned i = 0; i < instanced->count(); i++) {
                    unsigned index = i * 16;
                    tmp.fromArray(array, index);
                    tmp.premultiply(*obj.matrixWorld);
                    auto body = createSingle(tmp.elements.data());
                    bodies[&obj].emplace_back(body);
                }
            } else {
                bodies[&obj].emplace_back(createSingle(obj.matrixWorld->elements.data()));
            }
        }

        for (auto shape : shapes) {
            shape->release();
        }

        for (auto actor : bodies[&obj]) {
            scene->addActor(*actor);
        }

        obj.matrixAutoUpdate = false;
        obj.addEventListener("remove", &onMeshRemovedListener);
    }

    physx::PxJoint* createJoint(const JointCreate& create, threepp::Object3D* o1, threepp::Object3D* o2, threepp::Vector3 anchor, threepp::Vector3 axis) {

        o1->updateMatrixWorld();
        if (o2) o2->updateMatrixWorld();

        auto rb1 = getBody(*o1);
        auto rb2 = o2 ? getBody(*o2) : nullptr;

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

    physx::PxRigidActor* getBody(threepp::Object3D& mesh, size_t index = 0) {
        if (!bodies.count(&mesh)) return nullptr;
        return bodies.at(&mesh)[index];
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

        for (auto [_, list] : bodies) {
            for (auto rb : list) rb->release();
        }

        for (auto material : materials) material->release();
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
    std::vector<physx::PxMaterial*> materials;

    std::unordered_map<threepp::Object3D*, std::vector<physx::PxRigidActor*>> bodies;
    std::unordered_map<threepp::Object3D*, std::vector<physx::PxJoint*>> joints;

    std::shared_ptr<threepp::Mesh> debugTriangles = threepp::Mesh::create();
    std::shared_ptr<threepp::LineSegments> debugLines = threepp::LineSegments::create();
    std::shared_ptr<threepp::Points> debugPoints = threepp::Points::create();

    physx::PxMaterial* toPxMaterial(const std::optional<threepp::MaterialInfo> material) {
        if (!material) {
            return materials.front();
        }
        return physics->createMaterial(material->friction, material->friction, material->restitution);
    }

    physx::PxShape* toPxShape(const threepp::Collider& collider, physx::PxMaterial* material) {

        switch (collider.index()) {
            case 0:// Sphere
            {
                threepp::SphereCollider sphere = std::get<threepp::SphereCollider>(collider);
                return physics->createShape(physx::PxSphereGeometry(sphere.radius), *material, false);
            }
            case 1:// Box
            {
                threepp::BoxCollider box = std::get<threepp::BoxCollider>(collider);
                return physics->createShape(physx::PxBoxGeometry(box.halfWidth, box.halfHeight, box.halfDepth), *material, false);
            }
            case 2:// Capsule
            {
                threepp::CapsuleCollider box = std::get<threepp::CapsuleCollider>(collider);
                return physics->createShape(physx::PxCapsuleGeometry(box.radius, box.halfHeight), *material, false);
            }
            default:
                return nullptr;
        }
    }

    physx::PxShape* toPxShape(const threepp::BufferGeometry* geometry, physx::PxMaterial* material) {

        if (!geometry) return nullptr;

        const auto type = geometry->type();
        if (type == "BoxGeometry") {
            const auto box = dynamic_cast<const threepp::BoxGeometry*>(geometry);
            return physics->createShape(physx::PxBoxGeometry(physx::PxVec3{box->width / 2, box->height / 2, box->depth / 2}), *material, false);
        } else if (type == "SphereGeometry") {
            const auto sphere = dynamic_cast<const threepp::SphereGeometry*>(geometry);
            return physics->createShape(physx::PxSphereGeometry(sphere->radius), *material, false);
        } else if (type == "CapsuleGeometry") {
            const auto cap = dynamic_cast<const threepp::CapsuleGeometry*>(geometry);
            auto capShape = physics->createShape(physx::PxCapsuleGeometry(cap->radius, cap->length / 2), *material, false);
            capShape->setLocalPose(physx::PxTransform(physx::PxQuat(physx::PxHalfPi, {0, 0, 1})));
            return capShape;
        } else {

            if (auto pos = geometry->getAttribute<float>("position")) {
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

                return physics->createShape(physx::PxConvexMeshGeometry(convexMesh), *material, true);
            }

            return nullptr;
        }
    }

    void debugRender() {
        const auto& buffer = scene->getRenderBuffer();

        // lines

        std::vector<float> lineVertices(buffer.getNbLines() * 6);
        std::vector<float> lineColors(buffer.getNbLines() * 6);
        for (auto i = 0; i < buffer.getNbLines(); i++) {
            const auto& line = buffer.getLines()[i];

            lineVertices.emplace_back(line.pos0.x);
            lineVertices.emplace_back(line.pos0.y);
            lineVertices.emplace_back(line.pos0.z);
            lineVertices.emplace_back(line.pos1.x);
            lineVertices.emplace_back(line.pos1.y);
            lineVertices.emplace_back(line.pos1.z);

            threepp::Color c1 = line.color0;
            threepp::Color c2 = line.color1;
            lineColors.emplace_back(c1.r);
            lineColors.emplace_back(c1.g);
            lineColors.emplace_back(c1.b);
            lineColors.emplace_back(c2.r);
            lineColors.emplace_back(c2.g);
            lineColors.emplace_back(c2.b);
        }

        debugLines->visible = !lineVertices.empty();
        if (!lineVertices.empty()) {
            auto pos = debugLines->geometry()->getAttribute<float>("position");
            auto col = debugLines->geometry()->getAttribute<float>("color");
            if (pos && pos->array().size() >= lineVertices.size()) {//re-use geometry
                std::copy(lineVertices.begin(), lineVertices.end(), pos->array().begin());
                std::copy(lineColors.begin(), lineColors.end(), col->array().begin());
                pos->needsUpdate();
                col->needsUpdate();
                debugLines->geometry()->setDrawRange(0, lineVertices.size() / 3);
            } else {
                auto geom = threepp::BufferGeometry::create();
                geom->setAttribute("position", threepp::FloatBufferAttribute::create(lineVertices, 3));
                geom->setAttribute("color", threepp::FloatBufferAttribute::create(lineColors, 3));
                debugLines->setGeometry(geom);
            }
        }

        // triangles

        std::vector<float> triangleVertices(buffer.getNbTriangles() * 9);
        std::vector<float> triangleColors(buffer.getNbTriangles() * 9);
        for (auto i = 0; i < buffer.getNbTriangles(); i++) {
            const auto& point = buffer.getTriangles()[i];

            auto pos1 = point.pos0;
            auto pos2 = point.pos1;
            auto pos3 = point.pos2;

            triangleVertices.emplace_back(pos1.x);
            triangleVertices.emplace_back(pos1.y);
            triangleVertices.emplace_back(pos1.z);

            triangleVertices.emplace_back(pos2.x);
            triangleVertices.emplace_back(pos2.y);
            triangleVertices.emplace_back(pos2.z);

            triangleVertices.emplace_back(pos3.x);
            triangleVertices.emplace_back(pos3.y);
            triangleVertices.emplace_back(pos3.z);

            threepp::Color color1 = point.color0;
            threepp::Color color2 = point.color1;
            threepp::Color color3 = point.color2;

            triangleColors.emplace_back(color1.r);
            triangleColors.emplace_back(color1.g);
            triangleColors.emplace_back(color1.b);

            triangleColors.emplace_back(color2.r);
            triangleColors.emplace_back(color2.g);
            triangleColors.emplace_back(color2.b);

            triangleColors.emplace_back(color3.r);
            triangleColors.emplace_back(color3.g);
            triangleColors.emplace_back(color3.b);
        }

        debugTriangles->visible = !triangleVertices.empty();
        if (!triangleVertices.empty()) {
            auto pos = debugTriangles->geometry()->getAttribute<float>("position");
            auto col = debugTriangles->geometry()->getAttribute<float>("color");
            if (pos && pos->array().size() >= triangleVertices.size()) {//re-use geometry
                std::copy(triangleVertices.begin(), triangleVertices.end(), pos->array().begin());
                std::copy(triangleColors.begin(), triangleColors.end(), col->array().begin());
                pos->needsUpdate();
                col->needsUpdate();
                debugTriangles->geometry()->setDrawRange(0, triangleVertices.size() / 3);
            } else {
                auto geom = threepp::BufferGeometry::create();
                geom->setAttribute("position", threepp::FloatBufferAttribute::create(triangleVertices, 3));
                geom->setAttribute("color", threepp::FloatBufferAttribute::create(triangleColors, 3));
                debugTriangles->setGeometry(geom);
            }
        }

        // points

        std::vector<float> pointVertices(buffer.getNbPoints() * 3);
        std::vector<float> pointColors(buffer.getNbPoints() * 3);
        for (auto i = 0; i < buffer.getNbPoints(); i++) {
            const auto& point = buffer.getPoints()[i];

            auto pos = point.pos;
            pointVertices.emplace_back(pos.x);
            pointVertices.emplace_back(pos.y);
            pointVertices.emplace_back(pos.z);

            threepp::Color color = point.color;
            pointColors.emplace_back(color.r);
            pointColors.emplace_back(color.g);
            pointColors.emplace_back(color.b);
        }

        debugPoints->visible = !pointVertices.empty();
        if (!pointVertices.empty()) {
            auto pos = debugPoints->geometry()->getAttribute<float>("position");
            auto col = debugPoints->geometry()->getAttribute<float>("color");
            if (pos && pos->array().size() >= pointVertices.size()) {//re-use geometry
                std::copy(pointVertices.begin(), pointVertices.end(), pos->array().begin());
                std::copy(pointColors.begin(), pointColors.end(), col->array().begin());
                pos->needsUpdate();
                col->needsUpdate();
                debugPoints->geometry()->setDrawRange(0, pointVertices.size() / 3);
            } else {
                auto geom = threepp::BufferGeometry::create();
                geom->setAttribute("position", threepp::FloatBufferAttribute::create(pointVertices, 3));
                geom->setAttribute("color", threepp::FloatBufferAttribute::create(pointColors, 3));
                debugPoints->setGeometry(geom);
            }
        }
    }

    struct MeshRemovedListener: threepp::EventListener {

        explicit MeshRemovedListener(PxEngine* scope): scope(scope) {}

        void onEvent(threepp::Event& event) override {
            if (event.type == "remove") {
                auto m = static_cast<threepp::Object3D*>(event.target);
                if (scope->bodies.count(m)) {
                    auto rb = scope->bodies.at(m);
                    scope->scene->removeActor(*rb.front());
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
